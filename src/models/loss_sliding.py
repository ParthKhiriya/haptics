import torch
import torch.nn as nn
import torch.nn.functional as F

class SlidingTextureLoss(nn.Module):
    """
    Phase 2.3 Speckle-Free Spectral & Perceptual Envelope Loss for Controlled Sliding.
    Replaces noisy 2D STFT frame-bin differences with Multi-Resolution Welch PSD Loss,
    1D Spectral Wasserstein (CDF) Distance, Differentiable Centroid/Spread Loss,
    and Multi-Scale Envelope Correlation Loss.
    """
    def __init__(
        self,
        sr: int = 16000,
        f_min: float = 30.0,
        f_max: float = 800.0,
        w_psd: float = 4.0,
        w_centroid: float = 4.0,
        w_env: float = 3.5,
        w_stat: float = 2.0,
        w_div: float = 1.5,
    ):
        super().__init__()
        self.sr = sr
        self.f_min = f_min
        self.f_max = f_max
        self.w_psd = w_psd
        self.w_centroid = w_centroid
        self.w_env = w_env
        self.w_stat = w_stat
        self.w_div = w_div
        self.fft_sizes = (256, 512, 1024, 2048)
        self.env_windows = (64, 128, 256, 512, 1024)

    def _compute_welch_psd_and_moments(self, y: torch.Tensor, n_fft: int = 1024):
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=y.device, dtype=y.dtype)
        stft = torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True)
        power = stft.abs().pow(2).mean(dim=-1)  # Welch PSD [B, F]

        freqs = torch.linspace(0, self.sr / 2.0, power.size(-1), device=y.device, dtype=y.dtype)
        mask_tac = (freqs >= self.f_min) & (freqs <= self.f_max)
        mask_wide = (freqs >= 20.0) & (freqs <= 2000.0)

        p_tac = power[:, mask_tac] + 1e-8
        f_tac = freqs[mask_tac].unsqueeze(0)

        p_norm = p_tac / (p_tac.sum(dim=-1, keepdim=True) + 1e-8)
        cdf = torch.cumsum(p_norm, dim=-1)

        centroid = (p_norm * f_tac).sum(dim=-1)
        spread = torch.sqrt(((f_tac - centroid.unsqueeze(-1)).pow(2) * p_norm).sum(dim=-1) + 1e-6)

        log_p_tac = 10.0 * torch.log10(p_tac)
        max_tac = log_p_tac.max(dim=-1, keepdim=True)[0]
        log_p_tac = torch.maximum(log_p_tac, max_tac - 55.0)

        p_wide = power[:, mask_wide] + 1e-9
        log_p_wide = 10.0 * torch.log10(p_wide)
        max_wide = log_p_wide.max(dim=-1, keepdim=True)[0]
        log_p_wide = torch.maximum(log_p_wide, max_wide - 60.0)

        return p_tac, log_p_tac, log_p_wide, p_norm, cdf, centroid, spread

    def _multi_res_welch_loss(self, pred: torch.Tensor, target: torch.Tensor):
        total_psd_loss = 0.0
        total_moment_loss = 0.0

        for n_fft in self.fft_sizes:
            p_tac, log_p_tac, log_p_wide, p_norm, cdf_p, cent_p, spread_p = self._compute_welch_psd_and_moments(pred, n_fft)
            t_tac, log_t_tac, log_t_wide, t_norm, cdf_t, cent_t, spread_t = self._compute_welch_psd_and_moments(target, n_fft)

            # Passband LSD + Wideband Roll-off LSD (prevents >800Hz cliff)
            lsd_tac = torch.sqrt(F.mse_loss(log_p_tac, log_t_tac) + 1e-6) / 10.0
            lsd_wide = torch.sqrt(F.mse_loss(log_p_wide, log_t_wide) + 1e-6) / 15.0

            cos_lin = 1.0 - F.cosine_similarity(p_tac, t_tac, dim=-1).mean()
            cos_log = 1.0 - F.cosine_similarity(log_p_tac, log_t_tac, dim=-1).mean()

            total_psd_loss = total_psd_loss + lsd_tac + 0.4 * lsd_wide + cos_lin + 0.5 * cos_log

            # 1D Wasserstein (CDF) Distance + Centroid + Spread Loss
            emd_loss = F.l1_loss(cdf_p, cdf_t) * 2.5
            cent_loss = F.l1_loss(cent_p / self.f_max, cent_t / self.f_max)
            spr_loss = F.l1_loss(spread_p / self.f_max, spread_t / self.f_max)
            total_moment_loss = total_moment_loss + emd_loss + cent_loss + 0.5 * spr_loss

        n = len(self.fft_sizes)
        return total_psd_loss / n, total_moment_loss / n

    def _envelope_and_stats_loss(self, pred: torch.Tensor, target: torch.Tensor):
        env_loss = 0.0
        for w in self.env_windows:
            stride = max(w // 4, 8)
            env_p = torch.sqrt(F.avg_pool1d(pred.pow(2), kernel_size=w, stride=stride) + 1e-8)
            env_t = torch.sqrt(F.avg_pool1d(target.pow(2), kernel_size=w, stride=stride) + 1e-8)

            l1_env = F.l1_loss(env_p, env_t)
            vp = env_p - env_p.mean(dim=-1, keepdim=True)
            vt = env_t - env_t.mean(dim=-1, keepdim=True)
            corr = (vp * vt).sum(dim=-1) / (
                torch.sqrt((vp.pow(2).sum(dim=-1) + 1e-8) * (vt.pow(2).sum(dim=-1) + 1e-8))
            )
            env_loss = env_loss + l1_env + (1.0 - corr.mean())
        env_loss = env_loss / len(self.env_windows)

        rms_p = torch.sqrt(pred.pow(2).mean(dim=-1) + 1e-8)
        rms_t = torch.sqrt(target.pow(2).mean(dim=-1) + 1e-8)
        rms_loss = F.l1_loss(rms_p, rms_t)

        peak99_p = torch.quantile(pred.abs(), 0.995, dim=-1)
        peak99_t = torch.quantile(target.abs(), 0.995, dim=-1)
        peak_max_p = pred.abs().amax(dim=-1)
        peak_max_t = target.abs().amax(dim=-1)
        peak_loss = F.l1_loss(peak99_p, peak99_t) + 0.5 * F.l1_loss(peak_max_p, peak_max_t)

        crest_p = peak99_p / (rms_p + 1e-6)
        crest_t = peak99_t / (rms_t + 1e-6)
        crest_loss = F.l1_loss(crest_p, crest_t)

        stat_loss = rms_loss + peak_loss + 0.5 * crest_loss
        return env_loss, stat_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor, return_components: bool = False):
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)
        if target.dim() == 2:
            target = target.unsqueeze(1)

        p_flat = pred.squeeze(1)
        t_flat = target.squeeze(1)

        psd_loss, centroid_loss = self._multi_res_welch_loss(p_flat, t_flat)
        env_loss, stat_loss = self._envelope_and_stats_loss(pred, target)

        # Batch Spectral Diversity Loss at n_fft=1024
        if p_flat.size(0) > 1:
            _, _, _, p_norm, _, cent_p, _ = self._compute_welch_psd_and_moments(p_flat, n_fft=1024)
            _, _, _, t_norm, _, cent_t, _ = self._compute_welch_psd_and_moments(t_flat, n_fft=1024)
            dist_p = torch.cdist(p_norm, p_norm, p=1)
            dist_t = torch.cdist(t_norm, t_norm, p=1)
            div_loss = F.l1_loss(dist_p, dist_t) + F.l1_loss(
                cent_p.std() / self.f_max, cent_t.std() / self.f_max
            )
        else:
            div_loss = torch.tensor(0.0, device=pred.device)

        total_loss = (
            self.w_psd * psd_loss
            + self.w_centroid * centroid_loss
            + self.w_env * env_loss
            + self.w_stat * stat_loss
            + self.w_div * div_loss
        )

        if return_components:
            return total_loss, {
                "psd": psd_loss.item(),
                "cent": centroid_loss.item(),
                "env": env_loss.item(),
                "stat": stat_loss.item(),
                "div": div_loss.item(),
            }
        return total_loss

# Backwards compatibility alias
SlidingHapticLoss = SlidingTextureLoss

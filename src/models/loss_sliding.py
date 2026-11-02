import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingTextureLoss(nn.Module):
    """
    Phase 2.4 Publication-Grade Composite Loss for Controlled Sliding Synthesis.
    """
    def __init__(
        self,
        sr: int = 16000,
        f_min: float = 30.0,
        f_max: float = 800.0,
        w_psd: float = 5.0,
        w_centroid: float = 4.0,
        w_stft: float = 2.0,
        w_env: float = 5.0,
        w_stat: float = 2.5,
        w_phase: float = 2.0,
        w_div: float = 1.2,
        return_dict: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.sr = sr
        self.f_min = f_min
        self.f_max = f_max
        self.w_psd = w_psd
        self.w_centroid = w_centroid
        self.w_stft = w_stft
        self.w_env = w_env
        self.w_stat = w_stat
        self.w_phase = w_phase
        self.w_div = w_div
        self.return_dict = return_dict
        self.fft_sizes = (256, 512, 1024, 2048)
        self.env_windows = (32, 64, 128, 256, 512, 1024)

    def _exact_welch_psd_and_moments(self, y: torch.Tensor, nperseg: int = 1024, hop: int = 512):
        segments = y.unfold(dimension=-1, size=nperseg, step=hop)  # [B, N_segs, 1024]
        segments = segments - segments.mean(dim=-1, keepdim=True)
        win = torch.hann_window(nperseg, device=y.device, dtype=y.dtype).view(1, 1, -1)
        win_norm = win.pow(2).sum() * self.sr
        spec = torch.fft.rfft(segments * win, n=nperseg, dim=-1)
        psd_full = (spec.abs().pow(2) / win_norm).mean(dim=1)  # [B, 513]

        freqs = torch.linspace(0.0, self.sr / 2.0, psd_full.size(-1), device=y.device, dtype=y.dtype)
        mask = (freqs >= self.f_min) & (freqs <= self.f_max)

        p_tac = psd_full[:, mask] + 1e-10  # [B, 50]
        f_tac = freqs[mask].unsqueeze(0)   # [1, 50]

        p_norm = p_tac / (p_tac.sum(dim=-1, keepdim=True) + 1e-10)
        centroid = (p_norm * f_tac).sum(dim=-1)  # [B]
        spread = torch.sqrt(((f_tac - centroid.unsqueeze(-1)).pow(2) * p_norm).sum(dim=-1) + 1e-6)

        log_p_tac = 10.0 * torch.log10(p_tac)
        max_tac = log_p_tac.max(dim=-1, keepdim=True)[0]
        log_p_tac = torch.maximum(log_p_tac, max_tac - 60.0)

        p_full_floor = torch.maximum(psd_full, psd_full.max(dim=-1, keepdim=True)[0] * 1e-6 + 1e-12)
        log_p_full = 10.0 * torch.log10(p_full_floor)

        stopband_ratio = psd_full[:, ~mask].sum(dim=-1) / (psd_full.sum(dim=-1) + 1e-8)
        return p_tac, log_p_tac, log_p_full, p_norm, centroid, spread, stopband_ratio

    def _full_rfft_moments_and_tilt_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        L = pred.size(-1)
        mag_p = torch.fft.rfft(pred, n=L, dim=-1).abs() + 1e-8
        mag_t = torch.fft.rfft(target, n=L, dim=-1).abs() + 1e-8
        freqs = torch.linspace(0.0, self.sr / 2.0, mag_p.size(-1), device=pred.device, dtype=pred.dtype)

        mask = (freqs >= self.f_min) & (freqs <= self.f_max)
        mp_tac = mag_p[:, mask]
        mt_tac = mag_t[:, mask]
        f_tac = freqs[mask].unsqueeze(0)

        mp_norm = mp_tac / (mp_tac.sum(dim=-1, keepdim=True) + 1e-8)
        mt_norm = mt_tac / (mt_tac.sum(dim=-1, keepdim=True) + 1e-8)

        cent_p = (mp_norm * f_tac).sum(dim=-1)
        cent_t = (mt_norm * f_tac).sum(dim=-1)

        spread_p = torch.sqrt(((f_tac - cent_p.unsqueeze(-1)).pow(2) * mp_norm).sum(dim=-1) + 1e-6)
        spread_t = torch.sqrt(((f_tac - cent_t.unsqueeze(-1)).pow(2) * mt_norm).sum(dim=-1) + 1e-6)

        # Emphasize low-centroid (<260 Hz) surfaces by 2.2x to eliminate C1/C2 centroid over-prediction
        low_cent_weight = torch.where(cent_t < 260.0, 2.2, 1.0)
        cent_err = (low_cent_weight * (cent_p - cent_t).abs() / self.f_max).mean()
        spread_err = F.l1_loss(spread_p / self.f_max, spread_t / self.f_max)

        # Sub-band energy tilt matching (30-180 Hz vs 180-400 Hz vs 400-800 Hz)
        pow_p = mp_tac.pow(2)
        pow_t = mt_tac.pow(2)
        f_1d = freqs[mask]
        m_lo = f_1d < 180.0
        m_mid = (f_1d >= 180.0) & (f_1d < 400.0)
        m_hi = f_1d >= 400.0

        tot_p = pow_p.sum(dim=-1) + 1e-8
        tot_t = pow_t.sum(dim=-1) + 1e-8
        tilt_loss = (
            F.l1_loss(pow_p[:, m_lo].sum(dim=-1) / tot_p, pow_t[:, m_lo].sum(dim=-1) / tot_t)
            + F.l1_loss(pow_p[:, m_mid].sum(dim=-1) / tot_p, pow_t[:, m_mid].sum(dim=-1) / tot_t)
            + F.l1_loss(pow_p[:, m_hi].sum(dim=-1) / tot_p, pow_t[:, m_hi].sum(dim=-1) / tot_t)
        )
        return cent_err + 0.5 * spread_err + 0.7 * tilt_loss

    def _multi_res_stft_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = 0.0
        for n_fft in self.fft_sizes:
            hop = n_fft // 4
            win = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
            s_p = torch.stft(pred, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True).abs()
            s_t = torch.stft(target, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True).abs()

            freqs = torch.linspace(0.0, self.sr / 2.0, s_p.size(1), device=pred.device, dtype=pred.dtype)
            mask = (freqs >= 25.0) & (freqs <= 850.0)
            s_p_band = s_p[:, mask, :] + 1e-6
            s_t_band = s_t[:, mask, :] + 1e-6

            sc_loss = torch.norm(s_t_band - s_p_band, p="fro") / (torch.norm(s_t_band, p="fro") + 1e-6)
            mag_loss = F.l1_loss(torch.log(s_p_band), torch.log(s_t_band))
            total = total + sc_loss + mag_loss
        return total / len(self.fft_sizes)

    def _envelope_and_stats_loss(self, pred: torch.Tensor, target: torch.Tensor):
        env_loss = 0.0
        weight_sum = 0.0
        for w in self.env_windows:
            # Priority weight 2.2x on fine/meso evaluation scales (64 and 256) and 1.5x on 1024
            scale_w = 2.2 if w in (64, 256) else (1.5 if w == 1024 else 1.0)
            stride = 1 if w in (64, 256, 1024) else max(w // 4, 1)
            pad_l = w // 2
            pad_r = w - 1 - pad_l
            env_p = torch.sqrt(F.avg_pool1d(F.pad(pred.pow(2), (pad_l, pad_r), mode="reflect"), kernel_size=w, stride=stride) + 1e-8)
            env_t = torch.sqrt(F.avg_pool1d(F.pad(target.pow(2), (pad_l, pad_r), mode="reflect"), kernel_size=w, stride=stride) + 1e-8)

            l1_env = F.l1_loss(env_p, env_t)
            mod_depth_loss = F.l1_loss(env_p.std(dim=-1), env_t.std(dim=-1))

            step = max(w // 8, 1)
            if env_p.size(-1) > step:
                diff_env_loss = F.l1_loss(env_p[:, :, step:] - env_p[:, :, :-step], env_t[:, :, step:] - env_t[:, :, :-step])
            else:
                diff_env_loss = torch.tensor(0.0, device=pred.device)

            vp = env_p - env_p.mean(dim=-1, keepdim=True)
            vt = env_t - env_t.mean(dim=-1, keepdim=True)
            corr = (vp * vt).sum(dim=-1) / (
                torch.sqrt((vp.pow(2).sum(dim=-1) + 1e-8) * (vt.pow(2).sum(dim=-1) + 1e-8))
            )
            corr_loss = 1.0 - corr.mean()

            env_loss = env_loss + scale_w * (l1_env + 1.5 * mod_depth_loss + 1.0 * diff_env_loss + 1.4 * corr_loss)
            weight_sum += scale_w
        env_loss = env_loss / weight_sum

        rms_p = torch.sqrt(pred.pow(2).mean(dim=-1) + 1e-8)
        rms_t = torch.sqrt(target.pow(2).mean(dim=-1) + 1e-8)
        rms_loss = F.l1_loss(rms_p, rms_t)

        peak99_p = torch.quantile(pred.abs(), 0.99, dim=-1)
        peak99_t = torch.quantile(target.abs(), 0.99, dim=-1)
        peak_max_p = pred.abs().amax(dim=-1)
        peak_max_t = target.abs().amax(dim=-1)
        peak_loss = F.l1_loss(peak99_p, peak99_t) + 1.0 * F.l1_loss(peak_max_p, peak_max_t)

        crest_p = peak_max_p / (rms_p + 1e-6)
        crest_t = peak_max_t / (rms_t + 1e-6)
        crest_loss = F.l1_loss(crest_p, crest_t)

        stat_loss = 1.5 * rms_loss + peak_loss + 0.6 * crest_loss
        return env_loss, stat_loss

    def _phase_coherent_waveform_loss(self, p_flat: torch.Tensor, t_flat: torch.Tensor) -> torch.Tensor:
        vp = p_flat - p_flat.mean(dim=-1, keepdim=True)
        vt = t_flat - t_flat.mean(dim=-1, keepdim=True)
        pcc_full = (vp * vt).sum(dim=-1) / (
            torch.sqrt((vp.pow(2).sum(dim=-1) + 1e-8) * (vt.pow(2).sum(dim=-1) + 1e-8))
        )

        L = p_flat.size(-1)
        freqs = torch.linspace(0.0, self.sr / 2.0, L // 2 + 1, device=p_flat.device, dtype=p_flat.dtype)
        low_mask = ((freqs >= 30.0) & (freqs <= 320.0)).to(p_flat.dtype).unsqueeze(0)

        p_low = torch.fft.irfft(torch.fft.rfft(p_flat, n=L, dim=-1) * low_mask, n=L, dim=-1)
        t_low = torch.fft.irfft(torch.fft.rfft(t_flat, n=L, dim=-1) * low_mask, n=L, dim=-1)

        vpl = p_low - p_low.mean(dim=-1, keepdim=True)
        vtl = t_low - t_low.mean(dim=-1, keepdim=True)
        pcc_low = (vpl * vtl).sum(dim=-1) / (
            torch.sqrt((vpl.pow(2).sum(dim=-1) + 1e-8) * (vtl.pow(2).sum(dim=-1) + 1e-8))
        )
        return 0.55 * (1.0 - pcc_full.mean()) + 0.45 * (1.0 - pcc_low.mean())

    def forward(self, pred: torch.Tensor, target: torch.Tensor, return_components: bool | None = None):
        if return_components is None:
            return_components = self.return_dict

        if pred.dim() == 2:
            pred = pred.unsqueeze(1)
        if target.dim() == 2:
            target = target.unsqueeze(1)

        p_flat = pred.squeeze(1)
        t_flat = target.squeeze(1)

        # 1. Exact Welch PSD & Spectral Moments
        p_tac, log_p, log_p_full, p_norm, cent_p, spread_p, stop_p = self._exact_welch_psd_and_moments(p_flat)
        t_tac, log_t, log_t_full, t_norm, cent_t, spread_t, _ = self._exact_welch_psd_and_moments(t_flat)

        lsd_tac = torch.sqrt(F.mse_loss(log_p, log_t) + 1e-6) / 10.0
        lsd_full = torch.sqrt(F.mse_loss(log_p_full, log_t_full) + 1e-6) / 10.0

        cos_lin = 1.0 - F.cosine_similarity(p_tac, t_tac, dim=-1).mean()
        cos_norm = 1.0 - F.cosine_similarity(p_norm, t_norm, dim=-1).mean()

        # Peak-weighted linear PSD shape loss (10x weight on resonant modal peaks)
        peak_weights = 1.0 + 9.0 * (t_norm / (t_norm.amax(dim=-1, keepdim=True) + 1e-8))
        peak_psd_l1 = (peak_weights * (p_norm - t_norm).abs()).mean() * 12.0

        psd_loss = lsd_tac + 0.5 * lsd_full + 1.8 * cos_lin + cos_norm + peak_psd_l1 + 2.5 * stop_p.mean()

        # 2. Outlier-Weighted Dual-Domain Centroid, Spread & Spectral Tilt Loss
        low_w = torch.where(cent_t < 260.0, 2.2, 1.0)
        welch_moment_loss = (
            (low_w * (cent_p - cent_t).abs() / self.f_max).mean()
            + 0.5 * F.l1_loss(spread_p / self.f_max, spread_t / self.f_max)
        )
        rfft_moment_loss = self._full_rfft_moments_and_tilt_loss(p_flat, t_flat)
        centroid_loss = welch_moment_loss + rfft_moment_loss

        # 3. Multi-Resolution STFT Loss
        stft_loss = self._multi_res_stft_loss(p_flat, t_flat)

        # 4. Fine-Boosted Multi-Scale Envelope & Amplitude Statistics Loss
        env_loss, stat_loss = self._envelope_and_stats_loss(pred, target)

        # 5. Scale-Invariant Phase-Coherent Waveform Loss
        phase_loss = self._phase_coherent_waveform_loss(p_flat, t_flat)

        # 6. Batch Spectral Diversity Loss
        if p_flat.size(0) > 1:
            dist_p = torch.cdist(p_norm, p_norm, p=1)
            dist_t = torch.cdist(t_norm, t_norm, p=1)
            div_loss = F.l1_loss(dist_p, dist_t) + F.l1_loss(cent_p.std() / self.f_max, cent_t.std() / self.f_max)
        else:
            div_loss = torch.tensor(0.0, device=pred.device)

        total_loss = (
            self.w_psd * psd_loss
            + self.w_centroid * centroid_loss
            + self.w_stft * stft_loss
            + self.w_env * env_loss
            + self.w_stat * stat_loss
            + self.w_phase * phase_loss
            + self.w_div * div_loss
        )

        if return_components:
            loss_dict = {
                "total": total_loss.item(),
                "psd": psd_loss.item(),
                "centroid": centroid_loss.item(),
                "cent": centroid_loss.item(),
                "stft": stft_loss.item(),
                "env": env_loss.item(),
                "stat": stat_loss.item(),
                "phase": phase_loss.item(),
                "div": div_loss.item(),
            }
            return total_loss, loss_dict
        return total_loss


# Backwards compatibility alias
SlidingHapticLoss = SlidingTextureLoss

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


def torch_hilbert_envelope(x: torch.Tensor, eps: float = 1e-7) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable analytic signal & Hilbert amplitude envelope using FFT.
    Args:
        x: (B, T) or (B, 1, T) real tensor
    Returns:
        envelope: (B, T) instantaneous amplitude
        inst_phase: (B, T) instantaneous phase in radians
    """
    if x.ndim == 3:
        x = x.squeeze(1)
    B, T = x.shape
    X = torch.fft.fft(x, dim=-1)

    h = torch.zeros(T, device=x.device, dtype=x.dtype)
    if T % 2 == 0:
        h[0] = 1.0
        h[T // 2] = 1.0
        h[1 : T // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (T + 1) // 2] = 2.0

    analytic = torch.fft.ifft(X * h.unsqueeze(0), dim=-1)
    envelope = torch.sqrt(analytic.real ** 2 + analytic.imag ** 2 + eps)
    inst_phase = torch.atan2(analytic.imag + eps, analytic.real + eps)
    return envelope, inst_phase


class ShiftInvariantTimeLoss(nn.Module):
    """
    Evaluates L1, L2, Negative Pearson Correlation, and Negative SI-SDR over a small
    micro-shift tolerance window [-max_shift, +max_shift] using vectorized 1D unfolding.
    """
    def __init__(self, max_shift: int = 4, temperature: float = 0.08):
        super().__init__()
        self.max_shift = max_shift
        self.temperature = temperature

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        if pred.ndim == 3:
            pred = pred.squeeze(1)
        if target.ndim == 3:
            target = target.squeeze(1)

        B, T = target.shape
        K = self.max_shift
        if K <= 0 or T <= 2 * K + 8:
            l1 = F.l1_loss(pred, target)
            return {"si_l1": l1, "si_sdr_loss": l1, "corr_loss": l1}

        # Pad target by K on both sides and extract 2K+1 shifted windows of length T
        target_pad = F.pad(target.unsqueeze(1), (K, K), mode="replicate").squeeze(1)  # (B, T + 2K)
        target_shifts = target_pad.unfold(dimension=-1, size=T, step=1)  # (B, 2K+1, T)
        pred_exp = pred.unsqueeze(1)  # (B, 1, T)

        # Per-shift L1 error: (B, 2K+1)
        l1_per_shift = torch.mean(torch.abs(pred_exp - target_shifts), dim=-1)
        # Softmin weights across shifts
        weights = F.softmax(-l1_per_shift / self.temperature, dim=-1)  # (B, 2K+1)
        si_l1 = torch.mean(torch.sum(weights * l1_per_shift, dim=-1))

        # Best aligned target per sample for SI-SDR and Pearson r
        best_idx = torch.argmin(l1_per_shift, dim=-1)  # (B,)
        batch_indices = torch.arange(B, device=pred.device)
        target_aligned = target_shifts[batch_indices, best_idx, :]  # (B, T)

        # Zero-mean Pearson correlation loss
        p_zm = pred - torch.mean(pred, dim=-1, keepdim=True)
        t_zm = target_aligned - torch.mean(target_aligned, dim=-1, keepdim=True)
        r_num = torch.sum(p_zm * t_zm, dim=-1)
        r_den = torch.sqrt(torch.sum(p_zm ** 2, dim=-1) * torch.sum(t_zm ** 2, dim=-1) + 1e-8)
        corr_loss = torch.mean(1.0 - (r_num / r_den))

        # SI-SDR loss
        alpha = torch.sum(p_zm * t_zm, dim=-1, keepdim=True) / (
            torch.sum(t_zm ** 2, dim=-1, keepdim=True) + 1e-8
        )
        s_target = alpha * t_zm
        e_noise = p_zm - s_target
        si_sdr = 10.0 * torch.log10(
            (torch.sum(s_target ** 2, dim=-1) + 1e-8) / (torch.sum(e_noise ** 2, dim=-1) + 1e-8)
        )
        si_sdr_loss = -torch.mean(si_sdr.clamp(min=-30.0, max=40.0)) * 0.05

        return {
            "si_l1": si_l1,
            "corr_loss": corr_loss,
            "si_sdr_loss": si_sdr_loss,
        }


class TransientPhysicsLoss(nn.Module):
    """
    Enforces impact transient physics:
    1. Analytic Hilbert Envelope L1 & Log-Envelope L1
    2. First & Second Temporal Derivatives (Attack Slope da/dt & Jerk d^2a/dt^2)
    3. Schroeder Backward Cumulative Energy Decay Curve (EDC) in dB (Damping ratio matching)
    4. Peak Amplitude & Crest Factor Consistency
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def schroeder_edc_db(self, x: torch.Tensor) -> torch.Tensor:
        """Computes normalized backward cumulative energy decay curve in dB."""
        energy = x ** 2 + self.eps
        rev_cum = torch.flip(torch.cumsum(torch.flip(energy, dims=[-1]), dim=-1), dims=[-1])
        total_e = rev_cum[:, 0:1].clamp(min=self.eps)
        edc_db = 10.0 * torch.log10((rev_cum / total_e).clamp(min=1e-5, max=1.0))
        return edc_db

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        if pred.ndim == 3:
            pred = pred.squeeze(1)
        if target.ndim == 3:
            target = target.squeeze(1)

        # 1. Analytic Hilbert Envelope Loss
        env_p, _ = torch_hilbert_envelope(pred, eps=self.eps)
        env_t, _ = torch_hilbert_envelope(target, eps=self.eps)
        env_l1 = F.l1_loss(env_p, env_t) + 0.5 * F.l1_loss(
            torch.log(env_p + 1e-4), torch.log(env_t + 1e-4)
        )

        # 2. Attack Slope (First & Second Derivatives) weighted by target impact envelope
        impact_weight = (env_t / (torch.amax(env_t, dim=-1, keepdim=True) + self.eps)).detach()
        w1 = 1.0 + 3.0 * impact_weight[:, :-1]
        w2 = 1.0 + 3.0 * impact_weight[:, :-2]

        dp = torch.diff(pred, dim=-1)
        dt = torch.diff(target, dim=-1)
        d2p = torch.diff(dp, dim=-1)
        d2t = torch.diff(dt, dim=-1)

        deriv_loss = torch.mean(w1 * torch.abs(dp - dt)) + 0.5 * torch.mean(w2 * torch.abs(d2p - d2t))

        # 3. Schroeder Energy Decay Curve (EDC) Loss
        edc_p = self.schroeder_edc_db(pred)
        edc_t = self.schroeder_edc_db(target)
        edc_loss = F.l1_loss(edc_p, edc_t) * 0.05

        # 4. Peak Amplitude & Crest Factor Loss
        peak_p = torch.amax(torch.abs(pred), dim=-1) + self.eps
        peak_t = torch.amax(torch.abs(target), dim=-1) + self.eps
        rms_p = torch.sqrt(torch.mean(pred ** 2, dim=-1) + self.eps)
        rms_t = torch.sqrt(torch.mean(target ** 2, dim=-1) + self.eps)

        log_peak_loss = F.l1_loss(torch.log(peak_p), torch.log(peak_t))
        cf_p = torch.log(peak_p / rms_p)
        cf_t = torch.log(peak_t / rms_t)
        crest_loss = F.l1_loss(cf_p, cf_t)

        return {
            "env_loss": env_l1,
            "deriv_loss": deriv_loss,
            "edc_loss": edc_loss,
            "peak_crest_loss": log_peak_loss + crest_loss,
        }


class DualRegimeVibrotactileSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss tailored to vibrotactile impact transients:
    - Ultra-short windows [32, 64, 128] for <2ms attack sharpness
    - High-res modal windows [256, 512, 1024] for resonant ringing frequencies
    - Includes Instantaneous Frequency (IF) & Group Delay (GD) phase coherence
    """
    def __init__(
        self,
        fft_sizes: tuple[int, ...] = (32, 64, 128, 256, 512, 1024),
        hop_sizes: tuple[int, ...] = (8, 16, 32, 64, 128, 256),
        win_lengths: tuple[int, ...] = (32, 64, 128, 256, 512, 1024),
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        if pred.ndim == 3:
            pred = pred.squeeze(1)
        if target.ndim == 3:
            target = target.squeeze(1)

        T = pred.shape[-1]
        sc_loss_total = 0.0
        mag_loss_total = 0.0
        phase_loss_total = 0.0
        valid_scales = 0

        for n_fft, hop, win_len in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            if T < n_fft:
                continue
            valid_scales += 1
            window = torch.hann_window(win_len, device=pred.device, dtype=pred.dtype)

            S_p = torch.stft(
                pred, n_fft=n_fft, hop_length=hop, win_length=win_len,
                window=window, center=True, return_complex=True
            )
            S_t = torch.stft(
                target, n_fft=n_fft, hop_length=hop, win_length=win_len,
                window=window, center=True, return_complex=True
            )

            mag_p = torch.abs(S_p) + 1e-7
            mag_t = torch.abs(S_t) + 1e-7

            # Spectral convergence + Log magnitude L1
            sc = torch.norm(mag_t - mag_p, p="fro", dim=(-2, -1)) / (
                torch.norm(mag_t, p="fro", dim=(-2, -1)) + 1e-7
            )
            sc_loss_total = sc_loss_total + torch.mean(sc)
            mag_loss_total = mag_loss_total + F.l1_loss(torch.log(mag_p), torch.log(mag_t))

            # Energy-weighted Instantaneous Frequency & Group Delay cosine phase loss
            energy_mask = (mag_t / (torch.amax(mag_t, dim=(-2, -1), keepdim=True) + 1e-7)).detach()
            phase_p = torch.angle(S_p)
            phase_t = torch.angle(S_t)

            # Instantaneous frequency (time derivative of phase)
            if phase_p.shape[-1] > 1:
                if_p = torch.diff(phase_p, dim=-1)
                if_t = torch.diff(phase_t, dim=-1)
                if_err = torch.mean(energy_mask[:, :, :-1] * (1.0 - torch.cos(if_p - if_t)))
            else:
                if_err = 0.0

            # Group delay (frequency derivative of phase)
            if phase_p.shape[-2] > 1:
                gd_p = torch.diff(phase_p, dim=-2)
                gd_t = torch.diff(phase_t, dim=-2)
                gd_err = torch.mean(energy_mask[:, :-1, :] * (1.0 - torch.cos(gd_p - gd_t)))
            else:
                gd_err = 0.0

            phase_loss_total = phase_loss_total + 0.5 * (if_err + gd_err)

        denom = max(1, valid_scales)
        return {
            "stft_sc": sc_loss_total / denom,
            "stft_mag": mag_loss_total / denom,
            "stft_phase": phase_loss_total / denom,
        }


class PsychophysicalMechanoreceptorLoss(nn.Module):
    """
    Perceptual Vibrotactile Loss weighted by human cutaneous mechanoreceptor sensitivity:
    - Pacinian Corpuscles (PC): Bandpass 40 - 800 Hz, peak sensitivity at ~250 Hz
    - Meissner Corpuscles (RA): Bandpass 5 - 100 Hz, peak sensitivity at ~35 Hz
    """
    def __init__(self, fs: int = 16000):
        super().__init__()
        self.fs = fs

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        if pred.ndim == 3:
            pred = pred.squeeze(1)
        if target.ndim == 3:
            target = target.squeeze(1)

        T = pred.shape[-1]
        freqs = torch.fft.rfftfreq(T, d=1.0 / float(self.fs), device=pred.device)  # (F,)

        # Human Pacinian (PC) sensitivity curve (Gaussian in log-frequency peaking at 250 Hz)
        log_f = torch.log10(freqs.clamp(min=1.0))
        w_pc = torch.exp(-0.5 * ((log_f - math.log10(250.0)) / 0.35) ** 2)
        w_pc = w_pc * (freqs >= 30.0).float() * (freqs <= 900.0).float()

        # Human Meissner (RA) sensitivity curve (peaking at 35 Hz)
        w_ra = torch.exp(-0.5 * ((log_f - math.log10(35.0)) / 0.30) ** 2)
        w_ra = w_ra * (freqs >= 5.0).float() * (freqs <= 120.0).float()

        P_f = torch.abs(torch.fft.rfft(pred, dim=-1)) + 1e-7
        T_f = torch.abs(torch.fft.rfft(target, dim=-1)) + 1e-7

        log_diff = torch.abs(torch.log(P_f) - torch.log(T_f))
        pc_loss = torch.mean(w_pc.unsqueeze(0) * log_diff)
        ra_loss = torch.mean(w_ra.unsqueeze(0) * log_diff)

        # Normalized Power Spectral Density (PSD) Jensen-Shannon Divergence
        psd_p = (P_f ** 2) / (torch.sum(P_f ** 2, dim=-1, keepdim=True) + 1e-8)
        psd_t = (T_f ** 2) / (torch.sum(T_f ** 2, dim=-1, keepdim=True) + 1e-8)
        m = 0.5 * (psd_p + psd_t)
        jsd = 0.5 * torch.sum(psd_p * torch.log((psd_p + 1e-8) / (m + 1e-8)), dim=-1) + \
              0.5 * torch.sum(psd_t * torch.log((psd_t + 1e-8) / (m + 1e-8)), dim=-1)
        psd_jsd_loss = torch.mean(jsd)

        return {
            "perceptual_pc_ra": pc_loss + ra_loss,
            "psd_jsd": psd_jsd_loss,
        }


class ControlledTapPublicationLoss(nn.Module):
    """
    Master composite loss function for Controlled-Tap Audio-to-Haptics.
    """
    def __init__(self, fs: int = 16000, max_shift_samples: int = 4):
        super().__init__()
        self.time_loss = ShiftInvariantTimeLoss(max_shift=max_shift_samples)
        self.physics_loss = TransientPhysicsLoss()
        self.stft_loss = DualRegimeVibrotactileSTFTLoss()
        self.perceptual_loss = PsychophysicalMechanoreceptorLoss(fs=fs)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        aux: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        t_dict = self.time_loss(pred, target)
        p_dict = self.physics_loss(pred, target)
        s_dict = self.stft_loss(pred, target)
        m_dict = self.perceptual_loss(pred, target)

        total = (
            2.0 * t_dict["si_l1"]
            + 1.0 * t_dict["corr_loss"]
            + 1.0 * t_dict["si_sdr_loss"]
            + 2.5 * p_dict["env_loss"]
            + 1.5 * p_dict["deriv_loss"]
            + 1.0 * p_dict["edc_loss"]
            + 1.0 * p_dict["peak_crest_loss"]
            + 1.5 * (s_dict["stft_sc"] + s_dict["stft_mag"])
            + 0.5 * s_dict["stft_phase"]
            + 2.0 * m_dict["perceptual_pc_ra"]
            + 1.5 * m_dict["psd_jsd"]
        )

        # Optional supervision on auxiliary envelope & log-peak gain heads
        if aux is not None and "pred_env" in aux:
            env_t, _ = torch_hilbert_envelope(target)
            aux_env_loss = F.l1_loss(aux["pred_env"].squeeze(1), env_t.detach())
            total = total + 0.5 * aux_env_loss

        metrics_log = {
            k: float(v.detach().cpu().item())
            for d in (t_dict, p_dict, s_dict, m_dict)
            for k, v in d.items()
        }
        metrics_log["total_reg_loss"] = float(total.detach().cpu().item())
        return total, metrics_log


# Backward compatibility alias
Sound2HapLossV2 = ControlledTapPublicationLoss

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SnakeActivation(nn.Module):
    """Periodic activation function (x + (1/a) * sin^2(a * x)) for tactile oscillation synthesis."""
    def __init__(self, channels: int, init_alpha: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1, channels, 1), init_alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha.clamp(min=0.1)
        return x + (1.0 / a) * torch.pow(torch.sin(a * x), 2)

class SEChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=-1)
        w = self.fc(w).unsqueeze(-1)
        return x * w

class DilatedResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.block = nn.Sequential(
            SnakeActivation(channels),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.GroupNorm(8, channels),
            SnakeActivation(channels),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.shortcut = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.block(x)

class AntiCheckerboardUpsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, scale_factor: int, kernel_size: int = 7):
        super().__init__()
        self.scale_factor = scale_factor
        pad = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.ReflectionPad1d(pad),
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size),
            nn.GroupNorm(8, out_ch),
        )
        self.res1 = DilatedResBlock(out_ch, kernel_size=3, dilation=1)
        self.res2 = DilatedResBlock(out_ch, kernel_size=3, dilation=3)
        self.res3 = DilatedResBlock(out_ch, kernel_size=3, dilation=9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="linear", align_corners=False)
        x = self.conv(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return x

class DifferentiableTactileFilterbank(nn.Module):
    """
    Coherent Multi-Band Tactile Synthesizer (40 log-spaced bands from 25 Hz to 1600 Hz).
    - Uses a deterministic broadband multi-sine/velvet basis + acoustic excitation seed instead of
      fresh per-step torch.randn() to eliminate STFT speckle gradient noise and envelope jitter.
    - Includes 8 roll-off bands (800-1600 Hz) shaped with a 4th-order Butterworth envelope
      to match the ground-truth filter roll-off and eliminate the >800 Hz PSD cliff.
    """
    def __init__(
        self,
        num_bands: int = 40,
        filter_len: int = 513,
        target_length: int = 4800,
        sr: int = 16000,
        f_min: float = 25.0,
        f_max: float = 1600.0,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.filter_len = filter_len
        self.target_length = target_length
        self.sr = sr

        freqs = torch.logspace(math.log10(f_min), math.log10(f_max), num_bands)
        self.register_buffer("center_freqs", freqs)

        # 4th-order Butterworth lowpass magnitude response at 800 Hz for natural >800Hz roll-off
        butter_gain = 1.0 / torch.sqrt(1.0 + torch.pow(freqs / 800.0, 8))
        self.register_buffer("butter_rolloff", butter_gain.view(1, num_bands, 1))

        t = torch.arange(-(filter_len // 2), filter_len // 2 + 1, dtype=torch.float32) / sr
        window = torch.hann_window(filter_len)

        filters = []
        for i in range(num_bands):
            fc = freqs[i].item()
            bw = max(fc * 0.16, 12.0)
            h = 2.0 * (bw / sr) * torch.sinc(bw * t) * torch.cos(2.0 * math.pi * fc * t)
            h = h * window
            h = h / (torch.norm(h, p=2) + 1e-8)
            filters.append(h)

        fir_bank = torch.stack(filters, dim=0).unsqueeze(1)  # [K, 1, filter_len]
        self.register_buffer("fir_bank", fir_bank)

        # Precompute a deterministic, uniform-envelope broadband excitation basis (seeded once)
        gen = torch.Generator()
        gen.manual_seed(2026)
        raw_basis = torch.randn(1, num_bands, target_length + filter_len - 1, generator=gen)
        band_basis = F.conv1d(raw_basis, fir_bank, groups=num_bands)  # [1, K, target_length]
        # Flatten local envelope fluctuations in the fixed basis so it has constant local RMS = 1.0
        local_pwr = F.avg_pool1d(band_basis.pow(2), kernel_size=129, stride=1, padding=64)
        band_basis = band_basis / torch.sqrt(local_pwr + 1e-6)
        band_basis = band_basis / (band_basis.std(dim=-1, keepdim=True) + 1e-6)
        self.register_buffer("coherent_basis", band_basis)

        # High-frequency bandpass filter (800-5500 Hz) to extract demodulated friction impulses from audio
        t_hf = torch.arange(-64, 65, dtype=torch.float32) / sr
        h_hf = (
            2.0 * (4000.0 / sr) * torch.sinc(4000.0 * t_hf) * torch.cos(2.0 * math.pi * 3000.0 * t_hf)
            * torch.hann_window(129)
        )
        h_hf = h_hf / (torch.norm(h_hf, p=2) + 1e-8)
        self.register_buffer("hf_filter", h_hf.view(1, 1, 129))

    def extract_audio_subbands(self, audio_03s: torch.Tensor):
        """
        Extracts band-decomposed acoustic excitations and multi-scale envelopes directly from raw 0.3s audio.
        audio_03s: [B, 1, 4800]
        """
        pad = self.filter_len // 2
        a_pad = F.pad(audio_03s, (pad, pad), mode="reflect")
        # Filter raw audio into the 40 tactile bands [B, K, 4800]
        audio_bands = F.conv1d(a_pad.expand(-1, self.num_bands, -1), self.fir_bank, groups=self.num_bands)
        audio_bands_norm = audio_bands / (audio_bands.std(dim=-1, keepdim=True) + 1e-5)

        # Also extract demodulated high-frequency friction envelope (800-5000 Hz rectified)
        a_hf_pad = F.pad(audio_03s, (64, 64), mode="reflect")
        hf_sig = F.conv1d(a_hf_pad, self.hf_filter).abs()  # [B, 1, 4800]

        # Low-frequency mechanical vibration energy (30-800 Hz sum)
        lf_sig = audio_bands[:, :32, :].pow(2).mean(dim=1, keepdim=True).sqrt()

        # Multi-scale smooth acoustic envelope [B, 2, 4800]
        env_lf = torch.sqrt(F.avg_pool1d(lf_sig.pow(2), kernel_size=129, stride=1, padding=64) + 1e-8)
        env_hf = torch.sqrt(F.avg_pool1d(hf_sig.pow(2), kernel_size=129, stride=1, padding=64) + 1e-8)
        env_lf = env_lf / (env_lf.mean(dim=-1, keepdim=True) + 1e-6)
        env_hf = env_hf / (env_hf.mean(dim=-1, keepdim=True) + 1e-6)

        return audio_bands_norm, torch.cat([env_lf, env_hf], dim=1)

    def forward(
        self,
        band_gains: torch.Tensor,
        time_mod: torch.Tensor,
        audio_03s: torch.Tensor | None = None,
        coherence_mix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        band_gains: [B, K, 1] material-specific spectral band weights (learned from audio spectrum + latents)
        time_mod: [B, K, L] fine temporal modulation
        """
        B, K, L = time_mod.shape
        basis = self.coherent_basis[:, :, :L].expand(B, -1, -1)

        if audio_03s is not None:
            audio_bands_norm, _ = self.extract_audio_subbands(audio_03s)
            alpha = coherence_mix if coherence_mix is not None else 0.45
            excitation = alpha * audio_bands_norm + torch.sqrt(1.0 - alpha ** 2 + 1e-6) * basis
        else:
            excitation = basis

        # Apply Butterworth roll-off above 800 Hz + learned per-band spectral gains + temporal envelope
        effective_gains = band_gains * self.butter_rolloff
        shaped = (excitation * effective_gains * time_mod).sum(dim=1, keepdim=True) / math.sqrt(K)
        return shaped

class SlidingSEANetDecoder(nn.Module):
    """
    Phase 2.3 Acoustic-Guided Coherent SEANet Decoder for Controlled Sliding Synthesis.
    Combines EnCodec latents with direct raw-audio spectral & envelope conditioning when available.
    """
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        target_length: int = 4800,
        num_tactile_bands: int = 40,
        upsample_rates: tuple = (5, 4, 4, 2),
        dropout: float = 0.1,
        output_dim: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.target_length = target_length
        self.num_tactile_bands = num_tactile_bands
        self.output_dim = output_dim

        # 1. Latent Processing (Preserves raw latent stats before InstanceNorm)
        self.latent_norm = nn.InstanceNorm1d(latent_dim, affine=True)
        self.channel_attn = SEChannelAttention(latent_dim, reduction=4)

        # 2. Direct Acoustic Spectral Encoder (64 log-power STFT bins from 0.3s audio + 256 latent stats)
        self.acoustic_spec_bins = 64
        cond_dim = latent_dim * 2 + self.acoustic_spec_bins
        self.global_cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.film_proj = nn.Linear(hidden_dim, hidden_dim * 2)

        # Direct Spectral Band Predictor (Maps global acoustic+latent conditioning -> 40 filterbank band gains)
        self.band_gain_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tactile_bands),
            nn.Softplus(),
        )

        # 3. Sequence Modeling Backbone
        self.input_proj = nn.Conv1d(latent_dim, hidden_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.lstm_norm = nn.LayerNorm(hidden_dim)

        # 4. Anti-Checkerboard Upsampling Backbone (30 -> 4800)
        self.upsample_blocks = nn.ModuleList()
        in_ch = hidden_dim
        for rate in upsample_rates:
            out_ch = max(in_ch // 2, 32)
            self.upsample_blocks.append(AntiCheckerboardUpsampleBlock(in_ch, out_ch, scale_factor=rate))
            in_ch = out_ch

        self.final_ch = in_ch

        # 5. Synthesis Heads
        self.filterbank = DifferentiableTactileFilterbank(
            num_bands=num_tactile_bands, target_length=target_length
        )

        # Fine temporal band-modulation head
        self.time_mod_head = nn.Sequential(
            SnakeActivation(self.final_ch),
            nn.Conv1d(self.final_ch, num_tactile_bands, kernel_size=7, padding=3),
            nn.Softplus(),
        )

        # Deterministic macro-groove waveform head
        self.det_head = nn.Sequential(
            SnakeActivation(self.final_ch),
            nn.Conv1d(self.final_ch, 1, kernel_size=7, padding=3),
        )

        # Acoustic Envelope Fusion Gate (combines learned envelope with raw audio low/high sub-band envelopes)
        self.env_fusion = nn.Sequential(
            nn.Conv1d(self.final_ch + 2, 32, kernel_size=15, padding=7),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=15, padding=7),
            nn.Softplus(),
        )

        # Global RMS & Mix Calibration Head
        self.calib_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def _extract_acoustic_spectrum(self, audio_03s: torch.Tensor | None, batch_size: int, device, dtype) -> torch.Tensor:
        """Computes a 64-bin log-power spectrum (20-4000 Hz) directly from the 0.3s audio."""
        if audio_03s is None:
            return torch.zeros(batch_size, self.acoustic_spec_bins, device=device, dtype=dtype)

        y = audio_03s.squeeze(1)  # [B, 4800]
        win = torch.hann_window(512, device=device, dtype=dtype)
        stft = torch.stft(y, n_fft=512, hop_length=128, win_length=512, window=win, return_complex=True)
        pwr = stft.abs().pow(2).mean(dim=-1)  # [B, 257]
        # Take first 64 bins (0 to ~2000 Hz at 31.25 Hz/bin)
        log_spec = torch.log10(pwr[:, 1 : self.acoustic_spec_bins + 1] + 1e-7)
        # Normalize per-sample mean & scale so spectral tilt and centroid are prominent
        log_spec = (log_spec - log_spec.mean(dim=-1, keepdim=True)) / (log_spec.std(dim=-1, keepdim=True) + 1e-5)
        return log_spec

    def forward(self, latents: torch.Tensor, audio: torch.Tensor | None = None) -> torch.Tensor:
        """
        latents: [B, 128, T] from EnCodec.
        audio: Optional [B, 1, 16000] or [B, 1, 4800] raw input audio for direct spectral/envelope guidance.
        """
        B = latents.size(0)
        if latents.size(-1) > 30:
            latents = latents[:, :, :23]

        audio_03s = None
        if audio is not None:
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)
            audio_03s = audio[:, :, : self.target_length]

        # 1. Global Conditioning from Raw Latents + Direct Acoustic Spectrum
        feat_mean = latents.mean(dim=-1)
        feat_std = latents.std(dim=-1) + 1e-6
        ac_spec = self._extract_acoustic_spectrum(audio_03s, B, latents.device, latents.dtype)
        global_cond = self.global_cond_mlp(torch.cat([feat_mean, feat_std, ac_spec], dim=-1))  # [B, hidden_dim]

        # 2. Latent Sequence Processing with FiLM Conditioning
        x = self.latent_norm(latents)
        x = self.channel_attn(x)
        x = F.interpolate(x, size=30, mode="linear", align_corners=False)
        x = self.input_proj(x)

        gamma, beta = torch.chunk(self.film_proj(global_cond).unsqueeze(-1), 2, dim=1)
        x = x * (1.0 + 0.5 * torch.tanh(gamma)) + 0.5 * beta

        x_seq = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x_seq)
        x = x + self.lstm_norm(lstm_out).transpose(1, 2)

        for block in self.upsample_blocks:
            x = block(x)

        if x.size(-1) != self.target_length:
            x = F.interpolate(x, size=self.target_length, mode="linear", align_corners=False)

        # 3. Coherent Multi-Band Synthesis
        band_gains = self.band_gain_predictor(global_cond).unsqueeze(-1) + 0.05  # [B, 40, 1]
        time_mod = self.time_mod_head(x) + 0.1
        time_mod = time_mod / (time_mod.mean(dim=-1, keepdim=True) + 1e-6)

        calib = self.calib_head(global_cond).unsqueeze(-1)  # [B, 3, 1] in (0, 1)
        coherence_mix = 0.25 + 0.55 * calib[:, 0:1, :]       # [0.25, 0.80]
        det_weight = 0.35 * calib[:, 1:2, :]                 # [0.0, 0.35]
        target_rms = 0.85 + 0.20 * calib[:, 2:3, :]          # [0.85, 1.05]

        y_stoch = self.filterbank(band_gains, time_mod, audio_03s=audio_03s, coherence_mix=coherence_mix)
        y_det = self.det_head(x)

        raw_vib = (1.0 - det_weight) * y_stoch + det_weight * y_det

        # 4. Direct Acoustic-Guided Envelope Modulation
        if audio_03s is not None:
            _, ac_envs = self.filterbank.extract_audio_subbands(audio_03s)  # [B, 2, 4800]
        else:
            ac_envs = torch.ones(B, 2, self.target_length, device=x.device, dtype=x.dtype)

        env_gate = self.env_fusion(torch.cat([x, ac_envs], dim=1))  # [B, 1, 4800]
        env_gate = env_gate / (torch.sqrt(env_gate.pow(2).mean(dim=-1, keepdim=True)) + 1e-6)

        # Normalize local carrier variance before applying the fused temporal envelope
        carrier_local_rms = torch.sqrt(F.avg_pool1d(raw_vib.pow(2), kernel_size=257, stride=1, padding=128) + 1e-6)
        carrier_whitened = raw_vib / (0.6 * carrier_local_rms + 0.4 * raw_vib.std(dim=-1, keepdim=True) + 1e-6)

        out = carrier_whitened * env_gate
        out = out - out.mean(dim=-1, keepdim=True)

        # Calibrate global RMS to target_rms (~0.95)
        current_rms = torch.sqrt(out.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        out = out * (target_rms / current_rms)

        return F.hardtanh(out, min_val=-6.0, max_val=6.0)

# Backwards compatibility alias
SlidingTactileDecoder = SlidingSEANetDecoder

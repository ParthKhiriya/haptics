import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act2 = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor):
        res = x
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return x + res


class TactileSEANetDecoder(nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 1, target_length: int = 16000):
        """
        Translates EnCodec latent embeddings into vibrotactile waveforms.
        
        Args:
            input_dim: EnCodec latent dimension (128).
            output_dim: Output channels (1 for single-axis).
            target_length: Output waveform length in samples (16000 = 1.0s at 16kHz).
        """
        super().__init__()
        self.target_length = target_length
        
        # Temporal processing with dropout
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=input_dim, 
            num_layers=2, batch_first=True, dropout=0.1
        )
        
        # Upsampling: 75 frames * (4 * 4 * 4 * 2) = 9600 frames
        channels = [128, 64, 32, 16, 8]
        upsample_ratios = [4, 4, 4, 2]
        
        self.upsample_blocks = nn.ModuleList()
        for i in range(len(upsample_ratios)):
            stride = upsample_ratios[i]
            self.upsample_blocks.append(nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ConvTranspose1d(
                    in_channels=channels[i], out_channels=channels[i+1], 
                    kernel_size=stride * 2, stride=stride, padding=stride // 2
                ),
                ResNetBlock(channels[i+1])
            ))
        
        # Final projection
        self.final_conv = nn.Conv1d(channels[-1], output_dim, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor):
        # Input shape: [Batch, 128, 75]
        
        # Prepare for LSTM ([Batch, Time, Channels])
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        
        # Residual upsampling
        for block in self.upsample_blocks:
            x = block(x)
            
        # Final channel projection
        x = self.final_conv(x)
        
        # Linear response across target range [-5, 5] with loose safety clamp
        x = F.hardtanh(x, min_val=-6.0, max_val=6.0)
        
        # Interpolate 9,600 frames to target length (16,000)
        x = F.interpolate(x, size=self.target_length, mode='linear', align_corners=False)
        
        return x


class SnakeBeta(nn.Module):
    """
    Learnable periodic activation function (BigVGAN / DAC style) parameterized in log-space:
    f(x) = x + (1 / (exp(beta) + eps)) * sin^2(exp(alpha) * x)
    """
    def __init__(self, channels: int, alpha_init: float = 1.0, beta_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1, channels, 1), math.log(alpha_init)))
        self.beta = nn.Parameter(torch.full((1, channels, 1), math.log(beta_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha).clamp(min=0.05, max=50.0)
        beta = torch.exp(self.beta).clamp(min=0.05, max=50.0)
        return x + (1.0 / (beta + 1e-7)) * torch.pow(torch.sin(alpha * x), 2)


class DampedModalResonatorBank(nn.Module):
    """
    Differentiable Physics-Informed Modal Resonator Bank for Controlled-Tap.
    Synthesizes K exponentially damped sinusoids excited by a learned onset trigger:
    h_k(t) = exp(-zeta_k * omega_k * t) * sin(omega_k * t + phi_k)
    """
    def __init__(
        self,
        latent_dim: int,
        num_modes: int = 16,
        fs: int = 16000,
        f_min: float = 15.0,
        f_max: float = 900.0,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.fs = fs

        # Initialize modal frequencies logarithmically across the vibrotactile band [15 Hz, 900 Hz]
        init_freqs = torch.logspace(math.log10(f_min), math.log10(f_max), num_modes)
        self.base_log_freq = nn.Parameter(torch.log(init_freqs))  # (K,)
        self.base_log_damp = nn.Parameter(torch.full((num_modes,), math.log(0.06)))  # (K,)

        # Context-dependent modal parameter predictor from pooled audio latent
        self.modal_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, num_modes * 4),  # delta_freq, delta_damp, amplitude, phase
        )
        nn.init.zeros_(self.modal_head[-1].weight)
        nn.init.zeros_(self.modal_head[-1].bias)

        # Frame-level excitation detector from latent sequence
        self.excitation_conv = nn.Sequential(
            nn.Conv1d(latent_dim, 64, kernel_size=3, padding=1),
            SnakeBeta(64),
            nn.Conv1d(64, 1, kernel_size=1),
        )

    def forward(self, z: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Args:
            z: Audio latent tensor of shape (B, C, T_latent)
            target_len: Output waveform length T
        Returns:
            modal_wave: Synthesized modal ringing waveform (B, 1, T)
        """
        B = z.shape[0]
        device = z.device

        # Pool latent to predict material-specific modal parameters
        z_pool = torch.mean(z, dim=-1) + torch.amax(z, dim=-1)  # (B, C)
        params = self.modal_head(z_pool).view(B, self.num_modes, 4)

        df = torch.tanh(params[:, :, 0]) * 0.35
        dd = torch.tanh(params[:, :, 1]) * 0.80
        amps = F.softplus(params[:, :, 2]) * 0.25  # (B, K)
        phases = torch.tanh(params[:, :, 3]) * math.pi  # (B, K)

        freqs = torch.exp(self.base_log_freq.unsqueeze(0) + df).clamp(10.0, self.fs * 0.48)  # (B, K)
        zetas = torch.exp(self.base_log_damp.unsqueeze(0) + dd).clamp(0.003, 0.45)  # (B, K)

        # Build impulse response bank of length L_ir (up to 100 ms or target_len)
        L_ir = min(target_len, int(0.10 * self.fs))
        t = torch.arange(L_ir, device=device, dtype=z.dtype).view(1, 1, L_ir) / float(self.fs)  # (1, 1, L_ir)

        omega = 2.0 * math.pi * freqs.unsqueeze(-1)  # (B, K, 1)
        zeta = zetas.unsqueeze(-1)  # (B, K, 1)
        phi = phases.unsqueeze(-1)  # (B, K, 1)
        amp = amps.unsqueeze(-1)  # (B, K, 1)

        decay = torch.exp(-zeta * omega * t)
        osc = torch.sin(omega * torch.sqrt(1.0 - zeta ** 2 + 1e-6) * t + phi)
        ir = torch.sum(amp * decay * osc, dim=1, keepdim=True)  # (B, 1, L_ir)

        # Upsample latent excitation to waveform resolution
        exc_low = self.excitation_conv(z)  # (B, 1, T_latent)
        exc = F.interpolate(exc_low, size=target_len, mode="linear", align_corners=False)
        exc = F.relu(exc - exc.mean(dim=-1, keepdim=True))  # Sparse positive impact drive

        # Fast FFT convolution per batch item
        n_fft = 1 << (target_len + L_ir - 1).bit_length()
        E_f = torch.fft.rfft(exc.squeeze(1), n=n_fft, dim=-1)
        H_f = torch.fft.rfft(ir.squeeze(1), n=n_fft, dim=-1)
        y_modal = torch.fft.irfft(E_f * H_f, n=n_fft, dim=-1)[:, :target_len]
        return y_modal.unsqueeze(1)


class TransientEnvelopeModulator(nn.Module):
    """
    Predicts fine-grained temporal envelope e(t) and log-peak scale g from audio latents,
    and applies transient-sharpening FiLM modulation to the SEANet output waveform.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.env_net = nn.Sequential(
            nn.Conv1d(latent_dim, 64, kernel_size=5, padding=2),
            SnakeBeta(64),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            SnakeBeta(32),
            nn.Conv1d(32, 2, kernel_size=3, padding=1),  # [gamma_env, beta_env]
        )
        # Initialize last layer near zero so initial modulation is identity
        nn.init.zeros_(self.env_net[-1].weight)
        nn.init.zeros_(self.env_net[-1].bias)

        # High-resolution transient attack sharpener
        self.sharpen_conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, padding=3),
            SnakeBeta(16),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.sharpen_conv[-1].weight)
        nn.init.zeros_(self.sharpen_conv[-1].bias)

        # Global log-peak amplitude predictor (decoupled impact hardness head)
        self.log_gain_head = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.log_gain_head[-1].weight)
        nn.init.zeros_(self.log_gain_head[-1].bias)

    def forward(
        self,
        y_base: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            y_base: Base waveform from SEANetDecoder (B, 1, T)
            z: Encoded audio representation (B, C, T_latent)
        Returns:
            y_mod: Transient-modulated waveform (B, 1, T)
            pred_env: Predicted temporal envelope (B, 1, T)
            pred_log_gain: Predicted log-peak gain (B, 1)
        """
        T = y_base.shape[-1]
        env_feat = self.env_net(z)  # (B, 2, T_latent)
        env_up = F.interpolate(env_feat, size=T, mode="linear", align_corners=False)

        gamma = F.softplus(env_up[:, 0:1, :] + 0.5413)  # Softplus(0.5413) ~= 1.0 at init
        beta = torch.tanh(env_up[:, 1:2, :]) * 0.1

        y_film = gamma * y_base + beta
        delta_sharp = self.sharpen_conv(torch.cat([y_base, gamma], dim=1))
        y_mod = y_film + delta_sharp

        z_stats = torch.cat([torch.mean(z, dim=-1), torch.amax(z, dim=-1)], dim=-1)
        pred_log_gain = self.log_gain_head(z_stats)  # (B, 1)

        return y_mod, gamma, pred_log_gain


class TransientSEANetDecoder(nn.Module):
    """
    Publication-grade wrapper around the baseline SEANetDecoder.
    - Preserves 100% state_dict compatibility with `checkpoints/best_sean_decoder.pt`
    - Injects SnakeBeta periodic residual refinement, TransientEnvelopeModulator (FiLM),
      and DampedModalResonatorBank.
    """
    def __init__(
        self,
        base_decoder: nn.Module,
        latent_dim: int = 128,
        fs_haptic: int = 16000,
        num_modes: int = 16,
    ):
        super().__init__()
        self.base_decoder = base_decoder
        self.envelope_modulator = TransientEnvelopeModulator(latent_dim=latent_dim)
        self.modal_bank = DampedModalResonatorBank(
            latent_dim=latent_dim,
            num_modes=num_modes,
            fs=fs_haptic,
        )
        self.modal_mix_alpha = nn.Parameter(torch.tensor(0.15))

    def forward(
        self,
        z: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # 1. Base SEANet synthesis
        y_base = self.base_decoder(z)
        if y_base.ndim == 2:
            y_base = y_base.unsqueeze(1)  # Ensure (B, 1, T)

        T = y_base.shape[-1]

        # 2. Envelope FiLM modulation & transient spike sharpening
        y_mod, pred_env, pred_log_gain = self.envelope_modulator(y_base, z)

        # 3. Physics-informed damped modal resonator synthesis
        y_modal = self.modal_bank(z, target_len=T)
        y_out = y_mod + torch.tanh(self.modal_mix_alpha) * y_modal

        if return_aux:
            return y_out, {
                "y_base": y_base,
                "y_modal": y_modal,
                "pred_env": pred_env,
                "pred_log_gain": pred_log_gain,
            }
        return y_out


def load_and_upgrade_checkpoint(
    base_decoder: nn.Module,
    checkpoint_path: str = "checkpoints/best_sean_decoder.pt",
    latent_dim: int = 128,
    fs_haptic: int = 16000,
    num_modes: int = 16,
    device: str = "cpu",
) -> TransientSEANetDecoder:
    """
    Loads existing `best_sean_decoder.pt` into `TransientSEANetDecoder` with warm-start.
    Supports both raw state_dict and dict with 'model_state_dict' / 'state_dict'.
    """
    upgraded = TransientSEANetDecoder(
        base_decoder=base_decoder,
        latent_dim=latent_dim,
        fs_haptic=fs_haptic,
        num_modes=num_modes,
    )
    
    ckpt_file = Path(checkpoint_path)
    if ckpt_file.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

        # Check if state_dict has base_decoder keys or direct TactileSEANetDecoder keys
        has_base_prefix = any(k.startswith("base_decoder.") for k in state_dict.keys())
        if has_base_prefix:
            upgraded.load_state_dict(state_dict, strict=False)
        else:
            # Load into base_decoder
            base_decoder.load_state_dict(state_dict, strict=False)
        print(f"[*] Successfully warm-started TransientSEANetDecoder from {checkpoint_path}")
    else:
        print(f"[!] Warning: Checkpoint not found at {checkpoint_path}. Initialized from scratch.")

    return upgraded.to(device)

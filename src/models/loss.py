import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


class TactileBandSTFTLoss(nn.Module):
    """STFT loss with emphasis on the 50-300 Hz tactile band."""
    
    def __init__(self, fft_size: int, hop_size: int, win_length: int,
                 sample_rate: int = 16000, low_hz: float = 50.0, high_hz: float = 300.0):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        
        freq_res = sample_rate / fft_size
        self.low_bin = max(1, int(low_hz / freq_res))
        self.high_bin = min(fft_size // 2, int(high_hz / freq_res) + 1)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        window = torch.hann_window(self.win_length, device=x.device)
        
        x_stft = torch.stft(x.squeeze(1), n_fft=self.fft_size, hop_length=self.hop_size,
                            win_length=self.win_length, window=window, return_complex=True)
        y_stft = torch.stft(y.squeeze(1), n_fft=self.fft_size, hop_length=self.hop_size,
                            win_length=self.win_length, window=window, return_complex=True)
        
        x_mag, y_mag = x_stft.abs(), y_stft.abs()
        
        # Full-band losses
        sc_full = torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + 1e-7)
        mag_full = F.l1_loss(torch.log(x_mag + 1e-7), torch.log(y_mag + 1e-7))
        
        # Tactile-band focused losses (50-300 Hz)
        x_band = x_mag[:, self.low_bin:self.high_bin, :]
        y_band = y_mag[:, self.low_bin:self.high_bin, :]
        sc_band = torch.norm(y_band - x_band, p="fro") / (torch.norm(y_band, p="fro") + 1e-7)
        mag_band = F.l1_loss(torch.log(x_band + 1e-7), torch.log(y_band + 1e-7))
        
        return sc_full + mag_full, sc_band + mag_band


class MultiScaleEnvelopeLoss(nn.Module):
    """RMS envelope matching at multiple time scales + Pearson correlation."""
    
    def __init__(self, frame_lengths=[64, 256, 1024]):
        super().__init__()
        self.frame_lengths = frame_lengths
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        total_loss = 0.0
        for fl in self.frame_lengths:
            stride = fl // 2
            x_env = torch.sqrt(F.avg_pool1d(x ** 2, kernel_size=fl, stride=stride) + 1e-8)
            y_env = torch.sqrt(F.avg_pool1d(y ** 2, kernel_size=fl, stride=stride) + 1e-8)
            
            # RMS envelope MSE
            total_loss += F.mse_loss(x_env, y_env)
            
            # Pearson correlation loss (1 - r)
            xf = x_env.reshape(x_env.size(0), -1)
            yf = y_env.reshape(y_env.size(0), -1)
            xc = xf - xf.mean(dim=1, keepdim=True)
            yc = yf - yf.mean(dim=1, keepdim=True)
            corr = (xc * yc).sum(dim=1) / (xc.norm(dim=1) * yc.norm(dim=1) + 1e-8)
            total_loss += (1.0 - corr.mean())
        
        return total_loss / len(self.frame_lengths)


class Sound2HapLossV2(nn.Module):
    """Redesigned composite loss for tactile signal generation."""
    
    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = sample_rate
        
        # Multi-resolution STFT with tactile band emphasis
        self.stft_losses = nn.ModuleList([
            TactileBandSTFTLoss(1024, 256, 1024, sample_rate),
            TactileBandSTFTLoss(512,  128, 512,  sample_rate),
            TactileBandSTFTLoss(256,  64,  256,  sample_rate),
        ])
        
        # Multi-scale envelope
        self.envelope = MultiScaleEnvelopeLoss(frame_lengths=[64, 256, 1024])
        
        # Mel spectrogram with CORRECT sample rate and tactile-focused range
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=64,
            f_min=20,
            f_max=2000,
        )
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor):
        # 1. Time-domain losses
        l_mse = F.mse_loss(predicted, target)
        l_l1 = F.l1_loss(predicted, target)
        
        # 2. Multi-resolution spectral loss
        l_stft_full, l_stft_band = 0.0, 0.0
        for stft_fn in self.stft_losses:
            full, band = stft_fn(predicted, target)
            l_stft_full += full
            l_stft_band += band
        l_stft_full /= len(self.stft_losses)
        l_stft_band /= len(self.stft_losses)
        
        # 3. Log-mel spectrogram loss
        pred_mel = self.mel_transform.to(predicted.device)(predicted.squeeze(1))
        tgt_mel = self.mel_transform.to(target.device)(target.squeeze(1))
        l_mel = F.l1_loss(torch.log(pred_mel + 1e-7), torch.log(tgt_mel + 1e-7))
        
        # 4. Multi-scale envelope loss
        l_env = self.envelope(predicted, target)
        
        # Combined weighted loss
        total_loss = (
            1.0 * l_mse +
            1.0 * l_l1 +
            1.0 * l_stft_full +
            3.0 * l_stft_band +
            2.0 * l_mel +
            5.0 * l_env
        )
        
        metrics = {
            'mse': l_mse.item(),
            'l1': l_l1.item(),
            'stft_full': l_stft_full.item(),
            'stft_band': l_stft_band.item(),
            'mel': l_mel.item(),
            'envelope': l_env.item(),
        }
        return total_loss, metrics
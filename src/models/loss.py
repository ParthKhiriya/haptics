import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

class STFTLoss(nn.Module):
    def __init__(self, fft_size: int, shift_size: int, win_length: int):
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        window = torch.hann_window(self.win_length, device=x.device)
        
        x_mag = torch.stft(x.squeeze(1), n_fft=self.fft_size, hop_length=self.shift_size, 
                           win_length=self.win_length, window=window, return_complex=True).abs()
        y_mag = torch.stft(y.squeeze(1), n_fft=self.fft_size, hop_length=self.shift_size, 
                           win_length=self.win_length, window=window, return_complex=True).abs()
        
        sc_loss = torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + 1e-7)
        mag_loss = F.l1_loss(torch.log(x_mag + 1e-7), torch.log(y_mag + 1e-7))
        
        return sc_loss, mag_loss

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[1024, 512, 256], hop_sizes=[256, 128, 64], win_lengths=[1024, 512, 256]):
        super().__init__()
        self.stft_losses = nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses.append(STFTLoss(fs, ss, wl))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        return (sc_loss + mag_loss) / len(self.stft_losses)

class AmplitudeLoss(nn.Module):
    def __init__(self, frame_length=256):
        super().__init__()
        self.frame_length = frame_length
        
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x_rms = F.avg_pool1d(x**2, kernel_size=self.frame_length, stride=self.frame_length)
        y_rms = F.avg_pool1d(y**2, kernel_size=self.frame_length, stride=self.frame_length)
        return F.mse_loss(x_rms, y_rms)

class Sound2HapLoss(nn.Module):
    def __init__(self, sample_rate=10000):
        super().__init__()
        self.mse = nn.MSELoss()
        self.stft = MultiResolutionSTFTLoss()
        self.mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=80)
        self.amplitude = AmplitudeLoss()

    def forward(self, predicted: torch.Tensor, target: torch.Tensor):
        l_mse = self.mse(predicted, target)
        l_l1 = F.l1_loss(predicted, target) 
        l_stft = self.stft(predicted, target)
        
        window = torch.hann_window(1024, device=predicted.device)
        pred_mel = self.mel_transform.to(predicted.device)(predicted)
        target_mel = self.mel_transform.to(target.device)(target)
        l_mel = F.l1_loss(pred_mel, target_mel)
        
        l_amp = self.amplitude(predicted, target)
        
        # SANE WEIGHTS: Gentle guidance without exploding the gradients
        weight_mse = 10.0    
        weight_l1 = 2.0      
        weight_stft = 1.0    
        weight_mel = 1.0     
        weight_amp = 5.0    
        
        total_loss = (weight_mse * l_mse) + (weight_l1 * l_l1) + (weight_stft * l_stft) + (weight_mel * l_mel) + (weight_amp * l_amp)
        
        metrics = {
            'mse': l_mse.item(),
            'l1': l_l1.item(),
            'stft': l_stft.item(),
            'amp': l_amp.item()
        }
        return total_loss, metrics
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

class SlidingTactileDecoder(nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 1, target_length: int = 4800):
        """
        Translates EnCodec latent embeddings (from 1.0s continuous sliding audio)
        into high-fidelity sliding vibrotactile waveforms (0.3s at 16kHz = 4800 samples).
        
        Args:
            input_dim: EnCodec latent dimension (128).
            output_dim: Output channels (1 for 1D DFT321 acceleration).
            target_length: Output waveform length in samples (4800 = 0.3s at 16kHz).
        """
        super().__init__()
        self.target_length = target_length
        
        # Temporal processing to capture context across the full 1.0s sliding interaction
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=input_dim, 
            num_layers=2, batch_first=True, dropout=0.1
        )
        
        # Temporal compression: learns to compress 75 frames (1.0s context) -> 38 frames (~0.3s scale)
        self.temporal_compress = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            ResNetBlock(input_dim)
        )
        
        # Upsampling stages: 38 frames -> 152 -> 608 -> 2432 -> 4864 frames
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
        
        # Final channel projection
        self.final_conv = nn.Conv1d(channels[-1], output_dim, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor):
        # Input shape: [Batch, 128, 75]
        
        # 1. Temporal sequence modeling across full audio
        x = x.transpose(1, 2)  # [Batch, 75, 128]
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)  # [Batch, 128, 75]
        
        # 2. Compress audio context to haptic window scale
        x = self.temporal_compress(x)  # [Batch, 128, 38]
        
        # 3. Residual upsampling
        for block in self.upsample_blocks:
            x = block(x)
            
        # 4. Final projection
        x = self.final_conv(x)  # [Batch, 1, 4864]
        
        # 5. Linear response across target range [-5, 5] with headroom clamp
        x = F.hardtanh(x, min_val=-6.0, max_val=6.0)
        
        # 6. Interpolate to exact target length (4800 samples)
        x = F.interpolate(x, size=self.target_length, mode='linear', align_corners=False)
        
        return x

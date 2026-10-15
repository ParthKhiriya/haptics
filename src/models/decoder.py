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
        
        # Output bounded scaling: tanh constrains to [-1, 1], scaled by learnable factor
        self.output_scale = nn.Parameter(torch.tensor(3.0))

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
        
        # Bounded output activation
        x = torch.tanh(x) * self.output_scale
        
        # Interpolate 9,600 frames to target length (16,000)
        x = F.interpolate(x, size=self.target_length, mode='linear', align_corners=False)
        
        return x
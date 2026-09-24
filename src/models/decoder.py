import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        # Using LeakyReLU as per Sound2Hap specification
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act2 = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor):
        res = x
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return x + res

class TactileSEANetDecoder(nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 1, target_length: int = 10000):
        """
        Translates EnCodec latent embeddings into 10kHz vibrotactile waveforms.
        """
        super().__init__()
        self.target_length = target_length
        
        # 1. Temporal Processing: Two LSTM layers 
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=input_dim, 
                            num_layers=2, batch_first=True)
        
        # 2. Residual Upsampling Blocks
        # We upsample the 75 frames by a factor of 128 (4 * 4 * 4 * 2) = 9600 frames
        channels = [128, 64, 32, 16, 8]
        upsample_ratios = [4, 4, 4, 2]
        
        self.upsample_blocks = nn.ModuleList()
        for i in range(len(upsample_ratios)):
            stride = upsample_ratios[i]
            # Kernel = stride*2 and Padding = stride//2 guarantees exact multiplicative upsampling
            self.upsample_blocks.append(nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ConvTranspose1d(in_channels=channels[i], out_channels=channels[i+1], 
                                   kernel_size=stride * 2, stride=stride, padding=stride // 2),
                ResNetBlock(channels[i+1])
            ))
        
        # 3. Final Projection
        self.final_conv = nn.Conv1d(channels[-1], output_dim, kernel_size=7, padding=3)
        # Note: No Tanh activation is applied here, allowing unrestricted amplitude generation.

    def forward(self, x: torch.Tensor):
        # EnCodec Input: [Batch, 128, 75]
        
        # Prepare for LSTM (requires [Batch, Time, Channels])
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        
        # Revert back to [Batch, Channels, Time] for 1D Convolutions
        x = x.transpose(1, 2)
        
        # Pass through residual upsampling blocks
        for block in self.upsample_blocks:
            x = block(x)
            
        # Final channel projection to target dimension (e.g., 1 channel for Z-axis)
        x = self.final_conv(x)
        
        # Interpolate the 9,600 frames to exactly match the 10,000 sample target
        x = F.interpolate(x, size=self.target_length, mode='linear', align_corners=False)
        
        return x
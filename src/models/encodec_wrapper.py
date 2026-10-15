import torch
import torch.nn as nn
import torchaudio.transforms as T
from encodec import EncodecModel

class FrozenAudioEncoder(nn.Module):
    def __init__(self, target_bandwidth: float = 1.5, device: str = "cpu", input_sr: int = 16000):
        """
        Loads a pre-trained 24kHz EnCodec model and freezes its weights.
        
        Args:
            target_bandwidth: Bitrate for codebook selection.
            device: 'cpu' or 'cuda'.
            input_sr: Sample rate of input audio (from preprocessing).
                      Will be resampled to EnCodec's native 24kHz.
        """
        super().__init__()
        self.device = device
        self.input_sr = input_sr
        self.encodec_sr = 24000
        
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(target_bandwidth)
        self.model.to(self.device)
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        # Resample from ACTUAL working sample rate -> EnCodec's 24kHz
        self.resampler = T.Resample(
            orig_freq=self.input_sr,
            new_freq=self.encodec_sr
        ).to(self.device)

    @torch.no_grad()
    def forward(self, audio_tensor: torch.Tensor):
        """
        Input:  [B, 1, 16000] at 16kHz (1.0 second)
        Output: [B, 128, 75] continuous latent embeddings
        """
        # Resample 16kHz -> 24kHz for EnCodec
        audio_24k = self.resampler(audio_tensor)
        
        # Extract continuous latent embeddings (bypass quantizer)
        latent_embeddings = self.model.encoder(audio_24k)
        
        return latent_embeddings
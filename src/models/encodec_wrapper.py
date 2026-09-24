import torch
import torch.nn as nn
import torchaudio.transforms as T
from encodec import EncodecModel

class FrozenAudioEncoder(nn.Module):
    def __init__(self, target_bandwidth: float = 1.5, device: str = "cpu"):
        """
        Loads a pre-trained 24kHz EnCodec model and freezes its weights.
        target_bandwidth: Determines the bitrate and number of codebooks used.
        """
        super().__init__()
        self.device = device
        
        # Load the pre-trained 24kHz model
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(target_bandwidth)
        self.model.to(self.device)
        
        # Freeze all parameters to prevent backpropagation
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.eval()

        # Input audio is 10kHz, but EnCodec expects 24kHz.
        # We define a fast internal resampler for the forward pass.
        self.resampler = T.Resample(orig_freq=10000, new_freq=24000).to(self.device)

    @torch.no_grad()
    def forward(self, audio_tensor: torch.Tensor):
        """
        Passes 10kHz audio through the frozen encoder.
        Returns the continuous latent embeddings (not just the discrete integer codes) 
        so the tactile decoder has rich gradients to train on.
        """
        # 1. Resample to 24kHz
        # 1. Resample 10kHz audio to 24kHz for EnCodec
        audio_24k = self.resampler(audio_tensor)
        
        # 2. Extract continuous latent embeddings directly from the frozen encoder
        # Bypasses the lossy quantizer, preventing the codebook/batch-size collapse bug
        latent_embeddings = self.model.encoder(audio_24k)
        
        return latent_embeddings
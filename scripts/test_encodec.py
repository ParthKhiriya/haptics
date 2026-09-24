import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.encodec_wrapper import FrozenAudioEncoder

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading EnCodec on {device}...")
    
    encoder = FrozenAudioEncoder(target_bandwidth=1.5, device=device)
    
    # Simulate a single batch of audio from your DataLoader [Batch, Channels, Time]
    # 16 samples, 1 mono channel, 10,000 time steps (10kHz)
    dummy_audio_batch = torch.randn(16, 1, 10000).to(device)
    
    print("\nExtracting Latent Representations...")
    latent_embeddings = encoder(dummy_audio_batch)
    
    print("\nSuccess!")
    print(f"Input Audio Shape:      {dummy_audio_batch.shape}")
    print(f"Latent Embedding Shape: {latent_embeddings.shape}")
    print("\nNotice how the 10,000 time steps have been compressed into a dense, low frame-rate latent space.")

if __name__ == "__main__":
    main()
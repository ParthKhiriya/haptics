import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_dataloaders
from src.models.encodec_wrapper import FrozenAudioEncoder
from src.models.decoder import TactileSEANetDecoder

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    
    print("1. Loading DataLoader...")
    train_loader, _ = get_dataloaders(processed_dir, batch_size=4, window_size=10000)
    audio_batch, target_tactile_batch = next(iter(train_loader))
    audio_batch = audio_batch.to(device)
    
    print("2. Initializing Models...")
    encoder = FrozenAudioEncoder(target_bandwidth=1.5, device=device)
    
    # Check your tactile target channels (mono vs tri-axial)
    out_channels = target_tactile_batch.shape[1] 
    decoder = TactileSEANetDecoder(input_dim=128, output_dim=out_channels, target_length=10000).to(device)
    
    print("3. Forward Pass (Audio -> Latent -> Vibration)...")
    with torch.no_grad():
        latent_embeddings = encoder(audio_batch)
        predicted_tactile = decoder(latent_embeddings)
        
    print("\n--- Pipeline Success ---")
    print(f"Input Audio:         {audio_batch.shape}")
    print(f"Latent Bottleneck:   {latent_embeddings.shape}")
    print(f"Predicted Vibration: {predicted_tactile.shape}")
    print(f"Target Vibration:    {target_tactile_batch.shape}")

if __name__ == "__main__":
    main()
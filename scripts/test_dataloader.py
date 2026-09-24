import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_dataloaders

def main():
    processed_dir = PROJECT_ROOT / "data" / "processed"
    
    print("Initializing DataLoaders...")
    train_loader, val_loader = get_dataloaders(
        processed_dir=processed_dir,
        batch_size=16,         # 16 samples per batch
        window_size=10000,     # 1 second of data at 10kHz
        train_ratio=0.8
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Fetch a single batch to inspect shapes
    audio_batch, tactile_batch = next(iter(train_loader))
    
    print("\nBatch Shapes (Expected: [Batch, Channels, Time]):")
    print(f"Audio Tensor:   {audio_batch.shape}")
    print(f"Tactile Tensor: {tactile_batch.shape}")

if __name__ == "__main__":
    main()
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import numpy as np

class LMTHapticDataset(Dataset):
    def __init__(self, processed_dir: Path, window_size: int = 16000):
        """
        Args:
            processed_dir: Path to the normalized .npy files.
            window_size: Samples per window (16000 = 1.0s at 16kHz).
        """
        self.processed_dir = Path(processed_dir)
        self.window_size = window_size
        self.samples = []

        audio_files = sorted(self.processed_dir.rglob("*_audio.npy"))
        for aud_path in audio_files:
            tac_path = aud_path.parent / aud_path.name.replace("_audio.npy", "_tactile.npy")
            if tac_path.exists():
                self.samples.append((aud_path, tac_path))
                
        if len(self.samples) == 0:
            print(f"Warning: No paired samples found in {processed_dir}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        aud_path, tac_path = self.samples[idx]
        audio = np.load(aud_path).flatten()
        tactile = np.load(tac_path).flatten()

        min_len = min(len(audio), len(tactile))

        # Random cropping for data augmentation if array exceeds window_size
        if min_len > self.window_size:
            start = np.random.randint(0, min_len - self.window_size)
            audio = audio[start : start + self.window_size]
            tactile = tactile[start : start + self.window_size]
        else:
            audio = np.pad(audio[:min_len], (0, max(0, self.window_size - min_len)))
            tactile = np.pad(tactile[:min_len], (0, max(0, self.window_size - min_len)))
            
        # Per-window re-normalization for audio only (needed for EnCodec)
        audio = audio - audio.mean()
        peak = np.abs(audio).max() + 1e-8
        audio = audio / peak
        
        # Tactile is already z-scored and clipped at [-5, 5] during preprocessing.
        # Do NOT re-normalize here to avoid distorting target scale.
        tactile = np.asarray(tactile, dtype=np.float32)

        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        tactile_tensor = torch.from_numpy(tactile).float().unsqueeze(0)

        return audio_tensor, tactile_tensor

def get_dataloaders(processed_dir: Path, batch_size: int = 32, window_size: int = 16000, train_ratio: float = 0.8, num_workers: int = 0):
    """Creates Train and Validation DataLoaders."""
    dataset = LMTHapticDataset(processed_dir, window_size)
    
    if len(dataset) == 0:
        return None, None
    
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    # Generator seed fixed for reproducible splits
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    
    return train_loader, val_loader
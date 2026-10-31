import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import numpy as np

class LMTSlidingDataset(Dataset):
    def __init__(self, processed_dir: Path, audio_length: int = 16000, haptic_length: int = 4800):
        """
        Dataset for Controlled Sliding interactions.
        
        Args:
            processed_dir: Path to directory containing *_sliding_audio.npy and *_sliding_tactile.npy.
            audio_length: Samples for input audio window (default 16000 = 1.0s at 16kHz).
            haptic_length: Samples for target tactile window (default 4800 = 0.3s at 16kHz).
        """
        self.processed_dir = Path(processed_dir)
        self.audio_length = audio_length
        self.haptic_length = haptic_length
        self.samples = []

        audio_files = sorted(self.processed_dir.rglob("*_sliding_audio.npy"))
        for aud_path in audio_files:
            tac_path = aud_path.parent / aud_path.name.replace("_sliding_audio.npy", "_sliding_tactile.npy")
            if tac_path.exists():
                self.samples.append((aud_path, tac_path))

        if len(self.samples) == 0:
            print(f"Warning: No paired sliding samples found in {processed_dir}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        aud_path, tac_path = self.samples[idx]
        audio = np.load(aud_path).flatten()
        tactile = np.load(tac_path).flatten()

        # Enforce exact lengths
        if len(audio) > self.audio_length:
            audio = audio[:self.audio_length]
        elif len(audio) < self.audio_length:
            audio = np.pad(audio, (0, self.audio_length - len(audio)))

        if len(tactile) > self.haptic_length:
            tactile = tactile[:self.haptic_length]
        elif len(tactile) < self.haptic_length:
            tactile = np.pad(tactile, (0, self.haptic_length - len(tactile)))

        # Per-window audio re-normalization (required for EnCodec)
        audio = audio - audio.mean()
        peak = np.abs(audio).max() + 1e-8
        audio = audio / peak

        # Tactile is already z-scored and clipped at [-5, 5] during preprocessing
        tactile = np.asarray(tactile, dtype=np.float32)

        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)      # [1, 16000]
        tactile_tensor = torch.from_numpy(tactile).float().unsqueeze(0)   # [1, 4800]

        return audio_tensor, tactile_tensor

def get_sliding_dataloaders(processed_dir: Path, batch_size: int = 16,
                            audio_length: int = 16000, haptic_length: int = 4800,
                            train_ratio: float = 0.8, num_workers: int = 0):
    """Creates Train and Validation DataLoaders for Sliding data."""
    dataset = LMTSlidingDataset(processed_dir, audio_length=audio_length, haptic_length=haptic_length)

    if len(dataset) == 0:
        return None, None

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=num_workers
    )

    return train_loader, val_loader

import re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

VALID_STEM_PATTERN = re.compile(r"^C[1-8]_S\d+_M\d+_P\d+_X\d+_I\d+$")


class LMTSlidingDataset(Dataset):
    def __init__(
        self,
        samples: list,
        audio_length: int = 16000,
        haptic_length: int = 4800,
        is_train: bool = False,
    ):
        """
        Dataset for Controlled Sliding interactions.
        
        Args:
            samples: List of tuples (audio_path, haptic_path).
            audio_length: Samples for input audio window (default 16000 = 1.0s at 16kHz).
            haptic_length: Samples for target tactile window (default 4800 = 0.3s at 16kHz).
            is_train: Whether to apply JND-safe co-augmentation.
        """
        self.samples = samples
        self.audio_length = audio_length
        self.haptic_length = haptic_length
        self.is_train = is_train

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

        tactile = np.asarray(tactile, dtype=np.float32)

        # Subtle JND-safe haptic/audio co-augmentation during training
        if self.is_train:
            # 1. Random polarity flip (p=0.5) applied identically to both modalities
            if np.random.rand() < 0.5:
                audio = -audio
                tactile = -tactile
            # 2. Small amplitude jitter ~ U(0.92, 1.08) applied identically
            scale = np.random.uniform(0.92, 1.08)
            audio = np.clip(audio * scale, -1.0, 1.0)
            tactile = tactile * scale

        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)      # [1, 16000]
        tactile_tensor = torch.from_numpy(tactile).float().unsqueeze(0)   # [1, 4800]

        return audio_tensor, tactile_tensor


def discover_sliding_pairs(processed_dir: Path) -> list:
    """Discovers valid (audio, tactile) file pairs, strictly excluding any tmp* or malformed files."""
    processed_dir = Path(processed_dir)
    audio_files = sorted(processed_dir.rglob("*_sliding_audio.npy"))
    pairs = []
    for aud_path in audio_files:
        stem = aud_path.name.replace("_sliding_audio.npy", "")
        if stem.startswith("tmp") or not VALID_STEM_PATTERN.match(stem):
            continue

        # Look for _sliding_tactile.npy first, then fallback to _sliding_haptic.npy
        tac_path = aud_path.parent / f"{stem}_sliding_tactile.npy"
        if not tac_path.exists():
            tac_path = aud_path.parent / f"{stem}_sliding_haptic.npy"

        if tac_path.exists():
            pairs.append((aud_path, tac_path))

    return pairs


def get_stratified_split_indices(pairs: list, train_ratio: float = 0.8, seed: int = 42):
    """
    Performs deterministic stratified train/val splitting based on material class (C1–C8).
    Ensures all 8 classes are proportionally represented in train and val sets.
    """
    rng = np.random.RandomState(seed)
    class_to_indices = {}
    for idx, (aud_path, _) in enumerate(pairs):
        stem = aud_path.name.replace("_sliding_audio.npy", "")
        class_code = stem.split("_")[0] if "_" in stem else "Other"
        class_to_indices.setdefault(class_code, []).append(idx)

    train_indices = []
    val_indices = []

    for c_code in sorted(class_to_indices.keys()):
        c_idxs = np.array(class_to_indices[c_code])
        rng.shuffle(c_idxs)
        n_train = int(np.round(train_ratio * len(c_idxs)))
        # Ensure at least 1 validation sample if class has multiple samples
        if n_train == len(c_idxs) and len(c_idxs) > 1:
            n_train = len(c_idxs) - 1

        train_indices.extend(c_idxs[:n_train].tolist())
        val_indices.extend(c_idxs[n_train:].tolist())

    return sorted(train_indices), sorted(val_indices)


def get_sliding_splits(
    processed_dir: Path,
    audio_length: int = 16000,
    haptic_length: int = 4800,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    """Returns (train_dataset, val_dataset) with stratified split."""
    pairs = discover_sliding_pairs(processed_dir)
    if len(pairs) == 0:
        return None, None

    train_indices, val_indices = get_stratified_split_indices(pairs, train_ratio=train_ratio, seed=seed)
    train_samples = [pairs[i] for i in train_indices]
    val_samples = [pairs[i] for i in val_indices]

    train_dataset = LMTSlidingDataset(
        train_samples, audio_length=audio_length, haptic_length=haptic_length, is_train=True
    )
    val_dataset = LMTSlidingDataset(
        val_samples, audio_length=audio_length, haptic_length=haptic_length, is_train=False
    )
    return train_dataset, val_dataset


def get_sliding_dataloaders(
    processed_dir: Path,
    batch_size: int = 16,
    audio_length: int = 16000,
    haptic_length: int = 4800,
    train_ratio: float = 0.8,
    num_workers: int = 0,
    seed: int = 42,
):
    """Creates Train and Validation DataLoaders for Sliding data with stratified splitting."""
    train_dataset, val_dataset = get_sliding_splits(
        processed_dir,
        audio_length=audio_length,
        haptic_length=haptic_length,
        train_ratio=train_ratio,
        seed=seed,
    )
    if train_dataset is None or val_dataset is None:
        return None, None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers
    )

    return train_loader, val_loader

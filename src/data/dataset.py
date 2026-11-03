import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import numpy as np
from src.data.dataset_tap_utils import (
    compute_analytic_envelope,
    gcc_phat_envelope_lag,
    align_audio_and_haptic_tap,
    dft321_projection,
)

CLASS_MAP = {
    'C1': 0, 'C2': 1, 'C3': 2, 'C4': 3,
    'C5': 4, 'C6': 5, 'C7': 6, 'C8': 7,
}

class LMTHapticDataset(Dataset):
    def __init__(
        self,
        processed_dir: Path,
        window_size: int = 16000,
        micro_align: bool = True,
        return_aux: bool = False,
    ):
        """
        Args:
            processed_dir: Path to the normalized .npy files.
            window_size: Samples per window (16000 = 1.0s at 16kHz).
            micro_align: Whether to apply envelope GCC-PHAT micro-alignment.
            return_aux: If True, returns (audio, tactile, aux_dict); else (audio, tactile).
        """
        self.processed_dir = Path(processed_dir)
        self.window_size = window_size
        self.micro_align = micro_align
        self.return_aux = return_aux
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

        # Handle size matching
        if min_len > self.window_size:
            start = np.random.randint(0, min_len - self.window_size)
            audio = audio[start : start + self.window_size]
            tactile = tactile[start : start + self.window_size]
        else:
            audio = np.pad(audio[:min_len], (0, max(0, self.window_size - min_len)))
            tactile = np.pad(tactile[:min_len], (0, max(0, self.window_size - min_len)))

        # Sub-sample envelope micro-alignment if enabled
        if self.micro_align and len(audio) >= 256:
            env_h = compute_analytic_envelope(tactile, fs=16000, smooth_ms=0.4)
            env_a = compute_analytic_envelope(audio, fs=16000, smooth_ms=0.4)
            max_shift = int(0.003 * 16000)  # up to 3ms
            lag = gcc_phat_envelope_lag(env_h, env_a, max_shift_samples=max_shift)
            if lag > 0:
                audio = np.pad(audio[lag:], (0, lag), mode="constant")
            elif lag < 0:
                audio = np.pad(audio[:lag], (-lag, 0), mode="constant")

        # Per-window re-normalization for audio only (needed for EnCodec)
        audio = audio - audio.mean()
        peak_a = np.abs(audio).max() + 1e-8
        audio = audio / peak_a
        
        # Tactile is already z-scored and clipped at [-5, 5] during preprocessing
        tactile = np.asarray(tactile, dtype=np.float32)
        peak_h = float(np.max(np.abs(tactile)) + 1e-6)
        log_peak_gain = float(np.log(peak_h))

        # Extract material class index from filename
        stem = aud_path.stem.replace("_audio", "")
        c_code = stem.split("_")[0] if "_" in stem else "C1"
        class_idx = CLASS_MAP.get(c_code, 0)

        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        tactile_tensor = torch.from_numpy(tactile).float().unsqueeze(0)

        if self.return_aux:
            aux_dict = {
                "stem": stem,
                "class_code": c_code,
                "class_idx": class_idx,
                "log_peak_gain": log_peak_gain,
                "peak_gain": peak_h,
            }
            return audio_tensor, tactile_tensor, aux_dict

        return audio_tensor, tactile_tensor

def get_dataloaders(
    processed_dir: Path,
    batch_size: int = 32,
    window_size: int = 16000,
    train_ratio: float = 0.8,
    num_workers: int = 0,
    return_aux: bool = False,
):
    """Creates Train and Validation DataLoaders."""
    dataset = LMTHapticDataset(processed_dir, window_size=window_size, return_aux=return_aux)
    
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

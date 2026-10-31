import numpy as np
import scipy.io as sio
import librosa
from scipy.signal import butter, sosfiltfilt
from pathlib import Path
from tqdm import tqdm
import json
import warnings

warnings.filterwarnings("ignore")

# ─── SLIDING CONFIGURATION ────────────────────────────────────────
WORKING_SR = 16000          # Unified sample rate for the pipeline
AUDIO_NATIVE_SR = 44100     # sound (sliding) native sample rate
ACCEL_NATIVE_SR = 10000     # accelDFT native sample rate

AUDIO_DURATION = 1.0        # seconds — full continuous sliding audio
HAPTIC_DURATION = 0.3       # seconds — accelDFT recording duration

AUDIO_WINDOW_SAMPLES = int(WORKING_SR * AUDIO_DURATION)    # 16000 samples
HAPTIC_WINDOW_SAMPLES = int(WORKING_SR * HAPTIC_DURATION)  # 4800 samples

# Bandpass filter for tactile relevance (applied at native 10kHz)
# Extended to 1000 Hz for continuous sliding micro-textures
BANDPASS_LOW = 20           # Hz
BANDPASS_HIGH = 1000        # Hz
BANDPASS_ORDER = 4          # Butterworth filter order

# Z-score clipping to prevent gradient explosions
ZSCORE_CLIP = 5.0           # Clip at ±5 standard deviations
# ─────────────────────────────────────────────────────────────────


def _trim_or_pad(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Ensure signal length matches target_length exactly via center cropping or right padding."""
    n = len(signal)
    if n == target_length:
        return signal
    elif n > target_length:
        # Center crop
        start = (n - target_length) // 2
        return signal[start : start + target_length]
    else:
        # Pad with zeros at the end
        return np.pad(signal, (0, target_length - n))


class LMTSlidingPreprocessor:
    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Controlled Sliding field names
        self.audio_key = 'sound'
        self.accel_key = 'accelDFT'

        # Design bandpass filter once at native accelerometer sample rate (10 kHz)
        self.sos = butter(
            N=BANDPASS_ORDER,
            Wn=[BANDPASS_LOW, BANDPASS_HIGH],
            btype='bandpass',
            fs=ACCEL_NATIVE_SR,
            output='sos'
        )

    def _extract_sliding_pair(self, path: Path):
        """Extract continuous audio-haptic pair from controlled Sliding data."""
        mat = sio.loadmat(path)

        if 'finalMaterialRecording' not in mat:
            raise KeyError("Missing 'finalMaterialRecording' struct.")
        recording = mat['finalMaterialRecording'][0, 0]
        fields = recording.dtype.names

        # Verify sliding fields exist
        if self.audio_key not in fields or self.accel_key not in fields:
            raise KeyError(f"Missing sliding field(s). Available: {fields}")

        audio_raw = np.asarray(recording[self.audio_key], dtype=np.float32).flatten()
        accel_raw = np.asarray(recording[self.accel_key], dtype=np.float32).flatten()

        if audio_raw.size == 0 or accel_raw.size == 0:
            raise ValueError("Empty sliding data fields.")

        # ─── Step 1: Bandpass filter tactile signal at native 10 kHz ───
        accel_filtered = sosfiltfilt(self.sos, accel_raw).astype(np.float32)

        # ─── Step 2: Resample both to working sample rate (16 kHz) ───
        audio_16k = librosa.resample(audio_raw, orig_sr=AUDIO_NATIVE_SR, target_sr=WORKING_SR)
        accel_16k = librosa.resample(accel_filtered, orig_sr=ACCEL_NATIVE_SR, target_sr=WORKING_SR)

        # ─── Step 3: Exact shape enforcement ───
        audio_crop = _trim_or_pad(audio_16k, AUDIO_WINDOW_SAMPLES)
        accel_crop = _trim_or_pad(accel_16k, HAPTIC_WINDOW_SAMPLES)

        # ─── Step 4: Normalization ───
        # Audio: peak normalize to [-1, 1]
        audio_crop = audio_crop - audio_crop.mean()
        audio_crop = audio_crop / (np.max(np.abs(audio_crop)) + 1e-8)

        # Tactile: z-score normalize + clip extremes
        accel_crop = accel_crop - accel_crop.mean()
        accel_crop = accel_crop / (accel_crop.std() + 1e-8)
        accel_crop = np.clip(accel_crop, -ZSCORE_CLIP, ZSCORE_CLIP)

        return audio_crop.astype(np.float32), accel_crop.astype(np.float32)

    def process(self):
        mat_files = list(self.raw_dir.rglob("*.mat"))
        print(f"Discovered {len(mat_files)} .mat files. Processing controlled (Sliding) data...")

        saved, skipped, errors = 0, 0, 0
        error_log = []

        for file_path in tqdm(mat_files, desc="Processing Sliding"):
            try:
                audio, tactile = self._extract_sliding_pair(file_path)

                stem = file_path.stem
                np.save(self.processed_dir / f"{stem}_sliding_audio.npy", audio)
                np.save(self.processed_dir / f"{stem}_sliding_tactile.npy", tactile)
                saved += 1
            except KeyError as e:
                skipped += 1
                error_log.append({'file': file_path.name, 'error': str(e), 'type': 'missing_field'})
            except Exception as e:
                errors += 1
                error_log.append({'file': file_path.name, 'error': str(e), 'type': 'processing_error'})

        # Save processing report
        report = {
            'total_files': len(mat_files),
            'saved': saved,
            'skipped_missing_fields': skipped,
            'errors': errors,
            'config': {
                'working_sr': WORKING_SR,
                'audio_native_sr': AUDIO_NATIVE_SR,
                'accel_native_sr': ACCEL_NATIVE_SR,
                'bandpass': [BANDPASS_LOW, BANDPASS_HIGH],
                'bandpass_order': BANDPASS_ORDER,
                'audio_duration': AUDIO_DURATION,
                'haptic_duration': HAPTIC_DURATION,
                'audio_samples': AUDIO_WINDOW_SAMPLES,
                'haptic_samples': HAPTIC_WINDOW_SAMPLES,
                'zscore_clip': ZSCORE_CLIP,
                'audio_field': self.audio_key,
                'accel_field': self.accel_key,
            },
            'errors_detail': error_log
        }
        with open(self.processed_dir / 'processing_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nResults: {saved} saved, {skipped} skipped (missing fields), {errors} errors")
        print(f"Report saved to {self.processed_dir / 'processing_report.json'}")

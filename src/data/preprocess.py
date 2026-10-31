import numpy as np
import scipy.io as sio
import librosa
from scipy.signal import butter, sosfiltfilt
from pathlib import Path
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIGURATION ───────────────────────────────────────────────
WORKING_SR = 16000          # Unified sample rate for the pipeline
AUDIO_NATIVE_SR = 44100     # soundTap native sample rate  
ACCEL_NATIVE_SR = 10000     # accelTap native sample rate
WINDOW_DURATION = 1.0       # seconds — crop window around tap event
WINDOW_SAMPLES = int(WORKING_SR * WINDOW_DURATION)  # 16000 samples

# Bandpass filter for tactile relevance (applied at native 10kHz)
BANDPASS_LOW = 20           # Hz — below human tactile threshold
BANDPASS_HIGH = 500         # Hz — above Pacinian range but keeps some context
BANDPASS_ORDER = 4          # Butterworth filter order

# Z-score clipping to prevent gradient explosions
ZSCORE_CLIP = 5.0           # Clip at ±5 standard deviations

# Tap detection
PRE_TAP_MARGIN = 0.1        # seconds before peak to include
# ─────────────────────────────────────────────────────────────────


class LMTPreprocessor:
    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Controlled (Tap) field names
        self.audio_key = 'soundTap'
        self.accel_key = 'accelTap'
        
        # Design bandpass filter once (at native accel sample rate)
        self.sos = butter(
            N=BANDPASS_ORDER, 
            Wn=[BANDPASS_LOW, BANDPASS_HIGH],
            btype='bandpass', 
            fs=ACCEL_NATIVE_SR, 
            output='sos'
        )
    
    def _find_tap_peak(self, signal: np.ndarray) -> int:
        """Find the index of the tap event (maximum absolute amplitude)."""
        return int(np.argmax(np.abs(signal)))
    
    def _extract_aligned_pair(self, path: Path):
        """Extract temporally aligned audio-haptic pair from controlled Tap data."""
        mat = sio.loadmat(path)
        
        if 'finalMaterialRecording' not in mat:
            raise KeyError("Missing 'finalMaterialRecording' struct.")
        recording = mat['finalMaterialRecording'][0, 0]
        fields = recording.dtype.names
        
        # Verify controlled fields exist
        if self.audio_key not in fields or self.accel_key not in fields:
            raise KeyError(f"Missing controlled field(s). Available: {fields}")
        
        audio_raw = np.asarray(recording[self.audio_key], dtype=np.float32).flatten()
        accel_raw = np.asarray(recording[self.accel_key], dtype=np.float32).flatten()
        
        if audio_raw.size == 0 or accel_raw.size == 0:
            raise ValueError("Empty controlled data fields.")
        
        # ─── Step 1: Find tap event peaks in both signals ───
        audio_peak_idx = self._find_tap_peak(audio_raw)
        accel_peak_idx = self._find_tap_peak(accel_raw)
        
        audio_peak_time = audio_peak_idx / AUDIO_NATIVE_SR
        accel_peak_time = accel_peak_idx / ACCEL_NATIVE_SR
        
        # ─── Step 2: Bandpass filter tactile at native rate ───
        accel_filtered = sosfiltfilt(self.sos, accel_raw).astype(np.float32)
        
        # ─── Step 3: Resample both to working sample rate ───
        audio_16k = librosa.resample(audio_raw, orig_sr=AUDIO_NATIVE_SR, target_sr=WORKING_SR)
        accel_16k = librosa.resample(accel_filtered, orig_sr=ACCEL_NATIVE_SR, target_sr=WORKING_SR)
        
        # ─── Step 4: Peak-aligned cropping ───
        # Convert peak positions to working sample rate
        audio_peak_ws = int(audio_peak_time * WORKING_SR)
        accel_peak_ws = int(accel_peak_time * WORKING_SR)
        
        # Compute crop window: center on peak with small pre-tap margin
        pre_margin_samples = int(PRE_TAP_MARGIN * WORKING_SR)  # ~1600 samples
        
        # Audio crop: peak - margin to peak - margin + window
        a_start = max(0, audio_peak_ws - pre_margin_samples)
        a_end = a_start + WINDOW_SAMPLES
        if a_end > len(audio_16k):
            a_end = len(audio_16k)
            a_start = max(0, a_end - WINDOW_SAMPLES)
        
        # Accel crop: same temporal window relative to its own peak
        t_start = max(0, accel_peak_ws - pre_margin_samples)
        t_end = t_start + WINDOW_SAMPLES
        if t_end > len(accel_16k):
            t_end = len(accel_16k)
            t_start = max(0, t_end - WINDOW_SAMPLES)
        
        audio_crop = audio_16k[a_start:a_end]
        accel_crop = accel_16k[t_start:t_end]
        
        # Pad if shorter than window (shouldn't happen often)
        if len(audio_crop) < WINDOW_SAMPLES:
            audio_crop = np.pad(audio_crop, (0, WINDOW_SAMPLES - len(audio_crop)))
        if len(accel_crop) < WINDOW_SAMPLES:
            accel_crop = np.pad(accel_crop, (0, WINDOW_SAMPLES - len(accel_crop)))
        
        # ─── Step 5: Normalize ───
        # Audio: peak normalize to [-1, 1]
        audio_crop = audio_crop - audio_crop.mean()
        audio_crop = audio_crop / (np.max(np.abs(audio_crop)) + 1e-8)
        
        # Tactile: z-score normalize + clip extremes
        accel_crop = accel_crop - accel_crop.mean()
        accel_crop = accel_crop / (accel_crop.std() + 1e-8)
        accel_crop = np.clip(accel_crop, -ZSCORE_CLIP, ZSCORE_CLIP)
        
        return audio_crop, accel_crop
    
    def process(self):
        mat_files = list(self.raw_dir.rglob("*.mat"))
        print(f"Discovered {len(mat_files)} .mat files. Processing controlled (Tap) data...")
        
        saved, skipped, errors = 0, 0, 0
        error_log = []
        
        for file_path in tqdm(mat_files, desc="Processing"):
            try:
                audio, tactile = self._extract_aligned_pair(file_path)
                
                stem = file_path.stem
                np.save(self.processed_dir / f"{stem}_audio.npy", audio)
                np.save(self.processed_dir / f"{stem}_tactile.npy", tactile)
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
                'bandpass': [BANDPASS_LOW, BANDPASS_HIGH],
                'bandpass_order': BANDPASS_ORDER,
                'window_duration': WINDOW_DURATION,
                'window_samples': WINDOW_SAMPLES,
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
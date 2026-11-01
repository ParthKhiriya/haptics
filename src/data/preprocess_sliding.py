import re
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy import signal
from tqdm import tqdm

VALID_MAT_PATTERN = re.compile(r"^C[1-8]_S\d+_M\d+_P\d+_X\d+_I\d+$")

class SlidingPreprocessor:
    def __init__(
        self,
        raw_dir: str,
        processed_dir: str,
        audio_sr_orig: int = 44100,
        haptic_sr_orig: int = 10000,
        working_sr: int = 16000,
        audio_duration: float = 1.0,
        haptic_duration: float = 0.3,
        lowcut: float = 30.0,
        highcut: float = 800.0,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.audio_sr_orig = audio_sr_orig
        self.haptic_sr_orig = haptic_sr_orig
        self.working_sr = working_sr
        self.audio_len = int(audio_duration * working_sr)   # 16000
        self.haptic_len = int(haptic_duration * working_sr) # 4800
        self.lowcut = lowcut
        self.highcut = highcut

        self.sos_haptic = signal.butter(
            4, [self.lowcut, self.highcut], btype="bandpass", fs=self.working_sr, output="sos"
        )
        self.sos_audio_full = signal.butter(
            4, [25.0, 7600.0], btype="bandpass", fs=self.working_sr, output="sos"
        )
        self.sos_audio_low = signal.butter(
            4, [self.lowcut, self.highcut], btype="bandpass", fs=self.working_sr, output="sos"
        )
        self.sos_audio_high = signal.butter(
            4, [800.0, 6000.0], btype="bandpass", fs=self.working_sr, output="sos"
        )

    @staticmethod
    def _rms_env(x: np.ndarray, win: int) -> np.ndarray:
        kernel = np.ones(win, dtype=np.float64) / win
        return np.sqrt(np.convolve(x ** 2, kernel, mode="same") + 1e-8)

    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        return (x - np.mean(x)) / (np.std(x) + 1e-8)

    def _align_audio_window(
        self, audio_1s: np.ndarray, haptic_03s: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Aligns the 0.3s (4800-sample) haptic vibration against the 1.0s (16000-sample) audio
        using multi-scale (win=64 and win=256) dual-band (30-800Hz + 800-6000Hz) envelope
        cross-correlation at a fine 32-sample (2ms) step.
        """
        audio_low = signal.sosfiltfilt(self.sos_audio_low, audio_1s, padtype="odd", padlen=min(512, len(audio_1s) - 1))
        audio_high = signal.sosfiltfilt(self.sos_audio_high, audio_1s, padtype="odd", padlen=min(512, len(audio_1s) - 1))

        # Multi-scale target envelopes
        env_h_256 = self._zscore(self._rms_env(haptic_03s, 256))
        env_h_64 = self._zscore(self._rms_env(haptic_03s, 64))

        # Dual-band audio envelopes
        env_al_256 = self._rms_env(audio_low, 256)
        env_ah_256 = self._rms_env(audio_high, 256)
        env_al_64 = self._rms_env(audio_low, 64)
        env_ah_64 = self._rms_env(audio_high, 64)

        min_start = int(0.05 * self.working_sr)
        max_start = self.audio_len - self.haptic_len - int(0.05 * self.working_sr)
        stride = 32  # 2ms fine resolution

        best_score = -10.0
        best_start = int(0.35 * self.working_sr)

        for start in range(min_start, max_start + 1, stride):
            end = start + self.haptic_len
            al256 = self._zscore(env_al_256[start:end])
            ah256 = self._zscore(env_ah_256[start:end])
            al64 = self._zscore(env_al_64[start:end])
            ah64 = self._zscore(env_ah_64[start:end])

            comb256 = 0.65 * al256 + 0.35 * ah256
            comb64 = 0.65 * al64 + 0.35 * ah64

            c256 = np.mean(comb256 * env_h_256)
            c64 = np.mean(comb64 * env_h_64)
            score = 0.6 * c256 + 0.4 * c64

            if score > best_score:
                best_score = score
                best_start = start

        aligned_03s = audio_1s[best_start : best_start + self.haptic_len].copy()
        aligned_low = audio_low[best_start : best_start + self.haptic_len]

        # Align waveform polarity if low-band acoustic vibration is inverted relative to accelerometer
        if np.sum(aligned_low * haptic_03s) < 0:
            aligned_03s = -aligned_03s
            audio_1s = -audio_1s

        remaining = np.concatenate([audio_1s[:best_start], audio_1s[best_start + self.haptic_len :]])
        audio_reordered = np.concatenate([aligned_03s, remaining])[: self.audio_len]
        return audio_reordered.astype(np.float32), haptic_03s.astype(np.float32)

    def _extract_struct_array(self, mat: dict, keys: tuple) -> np.ndarray | None:
        # Check top-level dictionary first, then nested 'finalMaterialRecording' struct
        for k in keys:
            if k in mat:
                arr = np.asarray(mat[k], dtype=np.float64).squeeze()
                if arr.ndim == 1 and len(arr) > 500:
                    return arr
        if "finalMaterialRecording" in mat:
            rec = mat["finalMaterialRecording"]
            if isinstance(rec, np.ndarray) and rec.dtype.names is not None:
                rec_item = rec[0, 0] if rec.ndim == 2 else rec.item()
                for k in keys:
                    if k in rec.dtype.names:
                        arr = np.asarray(rec_item[k], dtype=np.float64).squeeze()
                        if arr.ndim == 1 and len(arr) > 500:
                            return arr
            elif isinstance(rec, dict):
                for k in keys:
                    if k in rec:
                        arr = np.asarray(rec[k], dtype=np.float64).squeeze()
                        if arr.ndim == 1 and len(arr) > 500:
                            return arr
        return None

    def process_file(self, mat_path: Path) -> bool:
        stem = mat_path.stem
        if not VALID_MAT_PATTERN.match(stem):
            return False

        try:
            mat = sio.loadmat(str(mat_path), simplify_cells=True)
        except Exception:
            return False

        raw_audio = self._extract_struct_array(mat, ("sound", "soundSteelTooltip"))
        raw_haptic = self._extract_struct_array(mat, ("accelDFT", "accelDFTSteelTooltip"))
        if raw_audio is None or raw_haptic is None:
            return False

        # 1. Process Audio: Detrend -> Polyphase Resample (44100 -> 16000) -> Bandpass
        raw_audio = signal.detrend(raw_audio, type="constant")
        audio_16k = signal.resample_poly(raw_audio, up=160, down=441, padtype="line")
        if len(audio_16k) >= self.audio_len:
            audio_16k = audio_16k[: self.audio_len]
        else:
            audio_16k = np.pad(audio_16k, (0, self.audio_len - len(audio_16k)), mode="reflect")

        audio_16k = signal.sosfiltfilt(
            self.sos_audio_full, audio_16k, padtype="odd", padlen=min(1024, len(audio_16k) - 1)
        )
        peak_a = np.percentile(np.abs(audio_16k), 99.8) + 1e-8
        audio_norm = np.clip(audio_16k / peak_a, -1.0, 1.0).astype(np.float32)

        # 2. Process Haptic (accelDFT): Detrend -> Mild Edge Taper -> Polyphase Resample -> Bandpass
        raw_haptic = signal.detrend(raw_haptic, type="linear")
        raw_haptic = raw_haptic - np.mean(raw_haptic)
        raw_haptic = raw_haptic * signal.windows.tukey(len(raw_haptic), alpha=0.04)

        haptic_16k = signal.resample_poly(raw_haptic, up=8, down=5, padtype="line")
        if len(haptic_16k) >= self.haptic_len:
            haptic_16k = haptic_16k[: self.haptic_len]
        else:
            haptic_16k = np.pad(haptic_16k, (0, self.haptic_len - len(haptic_16k)), mode="reflect")

        haptic_filtered = signal.sosfiltfilt(
            self.sos_haptic, haptic_16k, padtype="odd", padlen=min(900, len(haptic_16k) - 1)
        )
        haptic_filtered = haptic_filtered * signal.windows.tukey(self.haptic_len, alpha=0.03)
        haptic_filtered = haptic_filtered - np.mean(haptic_filtered)

        # Strict Unit-RMS = 1.0 Normalization with Soft Outlier Compression
        rms_h = np.sqrt(np.mean(haptic_filtered ** 2)) + 1e-8
        haptic_norm = haptic_filtered / rms_h
        # Soft-compress extreme spikes (>3.5 sigma) so they don't hit 5.0 and force RMS down
        haptic_norm = 4.2 * np.tanh(haptic_norm / 4.2)
        haptic_norm = haptic_norm - np.mean(haptic_norm)
        rms_post = np.sqrt(np.mean(haptic_norm ** 2)) + 1e-8
        haptic_norm = np.clip(haptic_norm / rms_post, -5.0, 5.0).astype(np.float32)

        # 3. Align 0.3s audio window to 0.3s haptic envelope
        audio_aligned, haptic_final = self._align_audio_window(audio_norm, haptic_norm)

        # Save audio and both naming variants of haptic for seamless backwards compatibility
        np.save(self.processed_dir / f"{stem}_sliding_audio.npy", audio_aligned)
        np.save(self.processed_dir / f"{stem}_sliding_haptic.npy", haptic_final)
        np.save(self.processed_dir / f"{stem}_sliding_tactile.npy", haptic_final)
        return True

    def run(self):
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        for stale in self.processed_dir.glob("tmp*"):
            stale.unlink(missing_ok=True)

        mat_files = sorted(self.raw_dir.rglob("*.mat"))
        valid_count = 0
        skipped_count = 0
        for mat_path in tqdm(mat_files, desc="Preprocessing Controlled Sliding"):
            if self.process_file(mat_path):
                valid_count += 1
            else:
                skipped_count += 1
        print(
            f"Successfully preprocessed {valid_count} valid sliding pairs into "
            f"{self.processed_dir} (skipped {skipped_count})"
        )

    def process(self):
        return self.run()

# Backwards compatibility alias
LMTSlidingPreprocessor = SlidingPreprocessor

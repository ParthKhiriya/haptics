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
        # Phase 2.4: Low macro-roughness band (30-350 Hz) for robust alignment & phase locking
        self.sos_align_low = signal.butter(
            4, [30.0, 350.0], btype="bandpass", fs=self.working_sr, output="sos"
        )

    @staticmethod
    def _rms_env(x: np.ndarray, win: int) -> np.ndarray:
        kernel = np.ones(win, dtype=np.float64) / win
        return np.sqrt(np.convolve(x ** 2, kernel, mode="same") + 1e-8)

    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        return (x - np.mean(x)) / (np.std(x) + 1e-8)

    @staticmethod
    def _fast_sliding_pcc(x_long: np.ndarray, y_short: np.ndarray, min_start: int, max_start: int) -> np.ndarray:
        """
        Computes exact Pearson correlation between y_short (length M) and every 1-sample slice
        x_long[s : s + M] for s in [min_start, max_start] in O(N log N) using FFT cross-correlation.
        """
        M = len(y_short)
        y_zm = y_short.astype(np.float64) - np.mean(y_short, dtype=np.float64)
        y_norm = np.linalg.norm(y_zm)
        if y_norm < 1e-8:
            return np.zeros(max_start - min_start + 1, dtype=np.float64)
        y_unit = y_zm / y_norm

        x_f = x_long.astype(np.float64)
        # Cross-correlation <x[s:s+M], y_unit> for all valid s
        cross = signal.correlate(x_f, y_unit, mode="valid", method="fft")[min_start : max_start + 1]

        # Exact sliding window mean and variance of x_f via cumulative sums
        cs1 = np.concatenate(([0.0], np.cumsum(x_f)))
        cs2 = np.concatenate(([0.0], np.cumsum(x_f * x_f)))
        s_idx = np.arange(min_start, max_start + 1)
        win_sum = cs1[s_idx + M] - cs1[s_idx]
        win_sq_sum = cs2[s_idx + M] - cs2[s_idx]
        win_var = np.maximum(win_sq_sum - (win_sum * win_sum) / M, 1e-12)
        return cross / np.sqrt(win_var)

    def _align_audio_window(
        self, audio_1s: np.ndarray, haptic_03s: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Band-Matched Vectorized 1-Sample Resolution Audio-Haptic Alignment:
        1. Filters audio_1s to the exact tactile passband (30-800 Hz) and low macro-roughness band (30-350 Hz)
           so high-frequency (1-7 kHz) air-hiss does not corrupt envelope or phase matching.
        2. Evaluates EVERY 1-sample offset in [0.06s, 0.66s] simultaneously via FFT cross-correlation across:
           - 64-sample (4ms), 256-sample (16ms), and 1024-sample (64ms) RMS envelopes
           - 30-800 Hz and 30-350 Hz waveform phase correlations (with optimal accelerometer polarity sign).
        """
        min_start = int(0.06 * self.working_sr)  # 960
        max_start = self.audio_len - self.haptic_len - int(0.04 * self.working_sr)  # 10560

        a_f64 = audio_1s.astype(np.float64)
        h_f64 = haptic_03s.astype(np.float64)

        # Bandpass audio to tactile band (30-800 Hz) and macro-roughness band (30-350 Hz)
        a_tac = signal.sosfiltfilt(self.sos_haptic, a_f64, padtype="odd", padlen=min(900, len(a_f64) - 1))
        a_low = signal.sosfiltfilt(self.sos_align_low, a_f64, padtype="odd", padlen=min(900, len(a_f64) - 1))
        h_low = signal.sosfiltfilt(self.sos_align_low, h_f64, padtype="odd", padlen=min(900, len(h_f64) - 1))

        # Mix 75% tactile-band power + 25% broadband power for physical envelope extraction
        a_pow = 0.75 * (a_tac / (np.std(a_tac) + 1e-8)) ** 2 + 0.25 * (a_f64 / (np.std(a_f64) + 1e-8)) ** 2
        h_pow = h_f64 ** 2

        def _rms_env(pow_sig: np.ndarray, win: int) -> np.ndarray:
            kernel = np.ones(win, dtype=np.float64) / float(win)
            return np.sqrt(np.convolve(pow_sig, kernel, mode="same") + 1e-10)

        c_env64 = self._fast_sliding_pcc(_rms_env(a_pow, 64), _rms_env(h_pow, 64), min_start, max_start)
        c_env256 = self._fast_sliding_pcc(_rms_env(a_pow, 256), _rms_env(h_pow, 256), min_start, max_start)
        c_env1024 = self._fast_sliding_pcc(_rms_env(a_pow, 1024), _rms_env(h_pow, 1024), min_start, max_start)

        c_wave_tac = self._fast_sliding_pcc(a_tac, h_f64, min_start, max_start)
        c_wave_low = self._fast_sliding_pcc(a_low, h_low, min_start, max_start)
        c_wave = 0.55 * c_wave_low + 0.45 * c_wave_tac

        # Joint multi-scale envelope + phase score at 1-sample resolution
        total_score = (
            0.32 * c_env64
            + 0.30 * c_env256
            + 0.14 * c_env1024
            + 0.24 * np.abs(c_wave)
        )
        best_idx = int(np.argmax(total_score))
        best_start = min_start + best_idx
        best_polarity = 1.0 if c_wave[best_idx] >= 0.0 else -1.0

        aligned_03s = audio_1s[best_start : best_start + self.haptic_len].copy()
        remaining = np.concatenate([audio_1s[:best_start], audio_1s[best_start + self.haptic_len :]])
        audio_reordered = np.concatenate([aligned_03s, remaining])[: self.audio_len]
        return audio_reordered.astype(np.float32), float(best_polarity)

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

        # 3. Align 0.3s audio window to 0.3s haptic envelope & lock phase polarity
        audio_aligned, polarity = self._align_audio_window(audio_norm, haptic_norm)
        haptic_final = (haptic_norm * polarity).astype(np.float32)

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
        for mat_path in tqdm(mat_files, desc="Preprocessing Controlled Sliding (Phase 2.4 Vectorized)"):
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

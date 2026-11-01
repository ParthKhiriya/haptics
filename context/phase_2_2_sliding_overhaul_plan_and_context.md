# Phase 2.2 — Controlled Sliding Audio-to-Tactile Synthesis Overhaul

## Part 1: Executive Summary of Forensic Diagnosis

An analysis of the repository documentation, evaluation metrics, and visual plots (`inference_sliding_result.jpg`, `evaluation_aggregate_psd.jpg`, `evaluation_envelopes.jpg`, `evaluation_spectral.jpg`, and `evaluation_report.json`) reveals why the current Controlled Sliding (`sound` $\to$ `accelDFT`) pipeline is underperforming.

Unlike Controlled Tap (`soundTap` $\to$ `accelTap`), which is a deterministic, phase-aligned impulse followed by damped sinusoidal ringing, **Controlled Sliding is a stationary/quasi-stationary stochastic frictional process** driven by microscopic stick-slip asperity collisions. Applying the tap-style deterministic pipeline and pointwise time-domain losses to sliding data has caused **five compounding failure modes**:

1. **Severe Spectral Mode Collapse (**$\sim 540\text{ Hz}$ **Centroid Lock):** Across all 201 validation samples in `evaluation_report.json`, the model predicts a locked spectral centroid between $535\text{ Hz}$ **and** $552\text{ Hz}$ **(mean** $\sim 540\text{ Hz}$**)**, regardless of whether the ground-truth centroid is $138.3\text{ Hz}$ (`C2_S3_M11`) or $1006.3\text{ Hz}$ (`C3_S1_M10`). The thin orange variance band in `evaluation_aggregate_psd.jpg` confirms the decoder outputs nearly the exact same average spectrum for every material.

2. **Periodic Carrier-Wave & Comb-Notch Artifacts:** In `inference_sliding_result.jpg`, the predicted waveform is a repetitive, amplitude-modulated sinusoidal wave packet oscillating at $\sim 540\text{ Hz}$ and modulated at $\sim 75\text{ Hz}$ (the exact EnCodec latent frame rate) rather than a stochastic frictional vibration. Strided `ConvTranspose1d` upsampling without stochastic noise injection creates severe high-frequency spectral notches above $700\text{ Hz}$ in `evaluation_aggregate_psd.jpg`.

3. **Destructive Loss Conflict (Time-Domain MSE/L1 vs. Spectral Loss):** Because `sound` and `accelDFT` are not phase-locked at the sub-millisecond level, pointwise time-domain MSE/L1 pushes the prediction toward zero ($\hat{y} \to 0$), resulting in a mean Pearson Correlation of $0.0004$, a negative SNR of $-1.71\text{ dB}$, and a compressed Peak Amplitude Ratio of $0.468\times$ (`Peak: 1.37` vs. `GT Peak: 5.00`).

4. **Temporal Duration Squishing (**$1.0\text{ s}$ **Audio** $\to 0.3\text{ s}$ **Haptic) & Edge-Transient Dominance:** Compressing $1.0\text{ s}$ of audio latents ($16,000$ samples) into $0.3\text{ s}$ of vibration ($4,800$ samples) warps temporal envelopes by $3.33\times$. Furthermore, FFT resampling and unpadded filtering in `preprocess_sliding.py` create massive boundary spikes at $t = 0.00\text{ s}$ and $t = 0.30\text{ s}$ (visible across `evaluation_envelopes.jpg` and at $t = 0.295\text{ s}$ in `inference_sliding_result.jpg`), causing the decoder to learn a flat horizontal line ($\sim 0.6$ RMS) with flared ends.

5. **Dataset Hygiene & Stopband LSD Blowup:** Corrupt temporary files (e.g., `tmpC1_S2_M62_I5_sliding_audio.npy` with class `"Unknown"`) leaked into the dataset, and full-band `log_spectral_distance` ($12.48\text{ dB}$ mean, up to $40.5\text{ dB}$) suffers from numerical log-floor blowups in the $800\text{--}8000\text{ Hz}$ stopband.

## Part 2: Antigravity Implementation Plan (Copy-Paste Prompt)

### 1. Objective & Workspace Context

You are working on the `controlled-sliding` pipeline of the **haptics** repository (Cross-Modal Audio-to-Tactile Synthesis on the TUM LMT Haptic Texture Database).
Depending on the active branch (`controlled-sliding` or `main`), the sliding modules are located either at:

* `src/data/preprocess_sliding.py` (or `src/data/preprocess.py` on the isolated branch)

* `src/data/dataset_sliding.py` (or `src/data/dataset.py`)

* `src/models/decoder_sliding.py` (or `src/models/decoder.py`)

* `src/models/loss_sliding.py` (or `src/models/loss.py`)

* `src/evaluation/metrics.py`

* `scripts/run_preprocessing_sliding.py`, `scripts/train_sliding.py`, `scripts/evaluate_sliding.py`, `scripts/inference_sliding.py`

**IMPORTANT HARNESS RULE:** First inspect the workspace directory (`ls src/data src/models scripts`) to verify whether the files use the `_sliding.py` suffix or standard names on the current branch, and apply all changes below to the corresponding active sliding files.

### 2. Root-Cause Diagnosis of Current Failure Modes

Current evaluation on $N = 201$ validation samples (`best_sliding_decoder.pt`) shows severe failure modes:

1. **Spectral Mode Collapse (**$\sim 540\text{ Hz}$ **Centroid Lock):** Every predicted sample has a spectral centroid locked between $535\text{--}552\text{ Hz}$ (`spectral_centroid_delta_hz` $= 115.8\text{ Hz}$) and `spectral_flatness_pred` locked at $\sim 2.2 \times 10^{-6}$, regardless of material class (`C1`–`C8`).

2. **Periodic Sinusoidal / Comb Artifacts:** Strided `ConvTranspose1d` upsampling from EnCodec frames generates a deterministic periodic carrier wave modulated at $75\text{ Hz}$ with comb notches above $700\text{ Hz}$ instead of realistic stochastic friction (`Crest Factor` $\sim 2.0$ vs. GT `Crest Factor` $\sim 4.5$, `peak_amplitude_ratio` $= 0.468$).

3. **Destructive Pointwise Time-Domain Loss:** `sound` and `accelDFT` are not sample-level phase-locked. Training with pointwise time-domain MSE/L1 on stochastic phase targets forces predictions toward the zero-mean expectation, causing `PCC` $= 0.0004$ and `SNR` $= -1.71\text{ dB}$.

4. $3.33\times$ **Temporal Compression & Filter Boundary Spikes:** Compressing $1.0\text{ s}$ of audio ($16,000$ samples) into $0.3\text{ s}$ of haptics ($4,800$ samples) distorts temporal envelopes by $3.33\times$. Furthermore, FFT resampling (`scipy.signal.resample`) on `accelDFT` introduces edge spikes at $t = 0.00\text{ s}$ and $t = 0.30\text{ s}$ that hijack the $[-5, 5]$ peak scaling and cause the model to predict a flat $0.6$ RMS line with U-shaped edge flares.

5. **Corrupt Temp Files & Stopband LSD Blowup:** Files starting with `tmp` (e.g., `tmpC1_S2_M62_I5.mat`) are ingested as class `"Unknown"`, and full-band `log_spectral_distance` blows up ($>35\text{ dB}$) due to unclamped $-\infty$ log power in the $800\text{--}8000\text{ Hz}$ stopband.

### 3. Step-by-Step Implementation Plan & Code Specifications

#### STEP 1: Overhaul Sliding Preprocessor (`src/data/preprocess_sliding.py` & `scripts/run_preprocessing_sliding.py`)

##### Changes Required:

1. **Strict Filename Regex Filter:** Ignore any `.mat` file whose stem does not match `^C[1-8]_S\d+_M\d+_P\d+_X\d+_I\d+$` (eliminates `tmpC1_...` corrupt files).

2. **Zero-Transient Polyphase Resampling & Edge Taper on `accelDFT`:**

   * Detrend `accelDFT` (`linear` detrend + mean subtraction).

   * Apply a $6\%$ Tukey window (`alpha=0.06`) to raw `accelDFT` *before* filtering and resampling to suppress window-edge step discontinuities at $t = 0.0\text{ s}$ and $t = 0.3\text{ s}$.

   * Filter with 4th-order Butterworth bandpass ($30\text{--}800\text{ Hz}$) using `sosfiltfilt(..., padtype='odd', padlen=min(900, len(x)-1))`.

   * Resample from $10,000\text{ Hz}$ ($3,000$ samples) to $16,000\text{ Hz}$ ($4,800$ samples) using `scipy.signal.resample_poly(x, up=8, down=5, padtype='line')` instead of FFT `resample`.

3. **1:1 Temporal Window Alignment (**$0.3\text{ s}$ **Audio Window Extraction):**

   * Instead of squishing $1.0\text{ s}$ of audio into $0.3\text{ s}$ of haptics ($3.33\times$ time distortion), extract the best-aligned $0.3\text{ s}$ ($4,800$ samples @ $16\text{ kHz}$) steady-state window from the $1.0\text{ s}$ audio using envelope cross-correlation against the $0.3\text{ s}$ `accelDFT` envelope (searching within $[0.10\text{ s}, 0.90\text{ s}]$ of the $1.0\text{ s}$ audio to avoid audio start/stop transients).

   * For backward compatibility with existing evaluation scripts: save `_sliding_audio.npy` as shape `(16000,)` where `audio[:4800]` is the exact envelope-aligned $0.3\text{ s}$ segment (and `audio[4800:]` is the remainder of the $1.0\text{ s}$ recording), and have `decoder_sliding.py` slice the aligned $0.3\text{ s}$ segment (`latents[:, :, :23]`) so `evaluation_report.json` still logs `"audio_length": 16000, "haptic_length": 4800` while the model operates on the exact 1:1 $0.3\text{ s}$ ($4,800$ sample) aligned audio.

4. **Robust Percentile/RMS Normalization:**

   * Scale `accelDFT` so its **RMS** $= 1.0$, and soft-clip extreme outliers using $5.0 \cdot \tanh(\text{haptic} / 5.0)$ so a single edge spike never crushes the steady-state texture RMS below $1.0$.

##### Reference Implementation for `src/data/preprocess_sliding.py`:

```
import os
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
        self.sos_audio = signal.butter(
            4, [20.0, 7800.0], btype="bandpass", fs=self.working_sr, output="sos"
        )

    def _align_audio_window(self, audio_1s: np.ndarray, haptic_03s: np.ndarray) -> np.ndarray:
        """
        Finds the 0.3s (4800-sample) window inside the 1.0s (16000-sample) audio
        whose broadband energy envelope best correlates with the 0.3s haptic envelope.
        Places the aligned 0.3s window in audio_out[:4800] for 1:1 temporal correspondence
        while keeping total array length 16000 for evaluation script compatibility.
        """
        win = 256
        kernel = np.ones(win, dtype=np.float32) / win
        env_a = np.sqrt(np.convolve(audio_1s ** 2, kernel, mode="same") + 1e-8)
        env_h = np.sqrt(np.convolve(haptic_03s ** 2, kernel, mode="same") + 1e-8)
        env_h_norm = (env_h - env_h.mean()) / (env_h.std() + 1e-8)

        # Search valid steady-state offsets between 0.1s (1600) and 0.65s (10400)
        min_start = int(0.10 * self.working_sr)
        max_start = self.audio_len - self.haptic_len - int(0.05 * self.working_sr)
        stride = 160  # 10ms step

        best_corr = -2.0
        best_start = int(0.35 * self.working_sr)  # Default to steady-state center
        for start in range(min_start, max_start + 1, stride):
            seg_env = env_a[start : start + self.haptic_len]
            seg_std = seg_env.std()
            if seg_std < 1e-6:
                continue
            corr = np.mean(((seg_env - seg_env.mean()) / seg_std) * env_h_norm)
            if corr > best_corr:
                best_corr = corr
                best_start = start

        aligned_03s = audio_1s[best_start : best_start + self.haptic_len].copy()
        # Construct 16000-length array where [:4800] is the aligned 0.3s window
        remaining = np.concatenate([audio_1s[:best_start], audio_1s[best_start + self.haptic_len :]])
        audio_reordered = np.concatenate([aligned_03s, remaining])[: self.audio_len]
        return audio_reordered.astype(np.float32)

    def process_file(self, mat_path: Path) -> bool:
        stem = mat_path.stem
        if not VALID_MAT_PATTERN.match(stem):
            return False

        try:
            mat = sio.loadmat(str(mat_path), simplify_cells=True)
        except Exception:
            return False

        sound_key = "sound" if "sound" in mat else ("soundSteelTooltip" if "soundSteelTooltip" in mat else None)
        accel_key = "accelDFT" if "accelDFT" in mat else ("accelDFTSteelTooltip" if "accelDFTSteelTooltip" in mat else None)
        if sound_key is None or accel_key is None:
            return False

        raw_audio = np.asarray(mat[sound_key], dtype=np.float64).squeeze()
        raw_haptic = np.asarray(mat[accel_key], dtype=np.float64).squeeze()
        if raw_audio.ndim != 1 or raw_haptic.ndim != 1 or len(raw_audio) < 1000 or len(raw_haptic) < 500:
            return False

        # 1. Process Audio: Detrend -> Polyphase Resample (44100 -> 16000) -> Bandpass
        raw_audio = signal.detrend(raw_audio, type="constant")
        audio_16k = signal.resample_poly(raw_audio, up=160, down=441, padtype="line")
        if len(audio_16k) >= self.audio_len:
            audio_16k = audio_16k[: self.audio_len]
        else:
            audio_16k = np.pad(audio_16k, (0, self.audio_len - len(audio_16k)), mode="reflect")

        audio_16k = signal.sosfiltfilt(self.sos_audio, audio_16k, padtype="odd", padlen=min(1024, len(audio_16k) - 1))

        # Normalize audio: preserve slight energy variation across surfaces while bounding to [-1, 1]
        peak_a = np.percentile(np.abs(audio_16k), 99.8) + 1e-8
        audio_norm = np.clip(audio_16k / peak_a, -1.0, 1.0).astype(np.float32)

        # 2. Process Haptic (accelDFT): Detrend -> Edge Taper -> Polyphase Resample (10000 -> 16000) -> Zero-Phase Bandpass
        raw_haptic = signal.detrend(raw_haptic, type="linear")
        raw_haptic = raw_haptic - np.mean(raw_haptic)
        # Apply 6% Tukey window BEFORE filtering/resampling to eliminate t=0.0s and t=0.3s step transients
        taper_orig = signal.windows.tukey(len(raw_haptic), alpha=0.06)
        raw_haptic = raw_haptic * taper_orig

        haptic_16k = signal.resample_poly(raw_haptic, up=8, down=5, padtype="line")
        if len(haptic_16k) >= self.haptic_len:
            haptic_16k = haptic_16k[: self.haptic_len]
        else:
            haptic_16k = np.pad(haptic_16k, (0, self.haptic_len - len(haptic_16k)), mode="reflect")

        haptic_filtered = signal.sosfiltfilt(
            self.sos_haptic, haptic_16k, padtype="odd", padlen=min(900, len(haptic_16k) - 1)
        )
        # Light post-filter edge taper (4%) so filter ringing at boundaries is zeroed
        haptic_filtered = haptic_filtered * signal.windows.tukey(self.haptic_len, alpha=0.04)
        haptic_filtered = haptic_filtered - np.mean(haptic_filtered)

        # Normalize Haptic to unit RMS = 1.0 with soft saturation at [-5.0, 5.0]
        rms_h = np.sqrt(np.mean(haptic_filtered ** 2)) + 1e-8
        haptic_norm = haptic_filtered / rms_h
        # Soft-clip outliers > 3.5 sigma into [-5.0, 5.0] without crushing RMS
        haptic_norm = 5.0 * np.tanh(haptic_norm / 5.0)
        # Re-adjust RMS to ~1.0 if soft-clipping reduced it, capping absolute peak at 5.0
        rms_post = np.sqrt(np.mean(haptic_norm ** 2)) + 1e-8
        scale_factor = min(1.0 / rms_post, 5.0 / (np.max(np.abs(haptic_norm)) + 1e-8))
        haptic_norm = (haptic_norm * scale_factor).astype(np.float32)

        # 3. Align 0.3s audio window to 0.3s haptic envelope
        audio_aligned = self._align_audio_window(audio_norm, haptic_norm)

        out_audio = self.processed_dir / f"{stem}_sliding_audio.npy"
        out_haptic = self.processed_dir / f"{stem}_sliding_haptic.npy"
        np.save(out_audio, audio_aligned)
        np.save(out_haptic, haptic_norm)
        return True

    def run(self):
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        # Remove any stale tmp* files from processed_dir
        for stale in self.processed_dir.glob("tmp*"):
            stale.unlink(missing_ok=True)

        mat_files = sorted(self.raw_dir.rglob("*.mat"))
        valid_count = 0
        for mat_path in tqdm(mat_files, desc="Preprocessing Controlled Sliding"):
            if self.process_file(mat_path):
                valid_count += 1
        print(f"Successfully preprocessed {valid_count} valid sliding pairs into {self.processed_dir}")

```

#### STEP 2: Update Dataset Loader (`src/data/dataset_sliding.py`)

##### Changes Required:

1. Filter out any `tmp*` or malformed `.npy` files when globbing `*_sliding_audio.npy`.

2. Ensure deterministic stratified splitting by **MaterialClass (`C1`–`C8`)** with fixed seed (`seed=42`) so all 8 classes are proportionally represented in both train and validation sets.

3. During training (`is_train=True`), apply subtle **JND-safe haptic/audio co-augmentation** (random polarity flip $p = 0.5$ on both audio and haptic, and small amplitude jitter $\text{scale} \sim \mathcal{U}(0.92, 1.08)$ applied identically to both audio and haptic) to prevent overfitting on the $\sim 900$ training instances.

#### STEP 3: Redesign the Sliding Decoder (`src/models/decoder_sliding.py`)

##### Why the Current Decoder Fails:

The current `SEANet` sliding decoder takes frozen EnCodec latents `[B, 128, 75]`, compresses them temporally, and upsamples via strided `ConvTranspose1d`. On stationary friction sounds:

* Frozen EnCodec embeddings have a dominant static channel bias and low inter-material variance.

* Strided `ConvTranspose1d` creates periodic $540\text{ Hz}$ sinusoidal ringing and comb notches.

* A purely deterministic network cannot synthesize realistic high-crest-factor ($\text{Peak}/\text{RMS} \sim 4.5$) stochastic surface vibration.

##### New Architecture (`SlidingSEANetDecoder`):

Keep the class name compatible with existing imports (`SlidingSEANetDecoder`) so `train_sliding.py`, `evaluate_sliding.py`, and `inference_sliding.py` work seamlessly.

1. **1:1 Aligned** $0.3\text{ s}$ **Audio Slicing:** When `latents` has $T = 75$ frames (from $1.0\text{ s}$ audio), slice `latents[:, :, :23]` corresponding to the aligned $0.3\text{ s}$ segment, eliminating the $3.33\times$ temporal compression bug.

2. **Latent Normalization & Channel Attention:** Apply `nn.InstanceNorm1d(128, affine=True)` + Squeeze-and-Excitation (`SEChannelAttention`) + global `FiLM` conditioning to strip out the static EnCodec bias and amplify material-discriminative channels.

3. **Anti-Checkerboard Upsampling (`AntiCheckerboardUpsampleBlock`):** Replace `nn.ConvTranspose1d` with `F.interpolate(..., mode='linear')` + `nn.ReflectionPad1d` + `nn.Conv1d` + **Snake Activation** ($x + \frac{1}{\alpha}\sin^2(\alpha x)$) + Dilated Residual Units.

4. **DDSP-Inspired Stochastic-Deterministic Synthesis Head:**

   * The decoder outputs two complementary streams at $16\text{ kHz}$ ($4,800$ samples):

     1. **Deterministic Structural Stream (`y_det`):** Captures macro-texture grooves, periodic ridges, and transient bumps.

     2. **Filtered Stochastic Micro-Asperity Stream (`y_stoch`):** Generates unit Gaussian white noise $\epsilon \sim \mathcal{N}(0, 1)$, shapes it in the frequency domain via a **Learned Multi-Band FIR Filterbank (32 log-spaced bands in** $30\text{--}800\text{ Hz}$**)** predicted from the latent representation, and modulates it in the time domain by learned high-resolution band envelopes.

##### Complete Drop-In Code for `src/models/decoder_sliding.py`:

```
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SnakeActivation(nn.Module):
    """Periodic activation function (x + (1/a) * sin^2(a * x)) for realistic oscillation & texture synthesis."""
    def __init__(self, channels: int, init_alpha: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1, channels, 1), init_alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha.clamp(min=0.1)
        return x + (1.0 / a) * torch.pow(torch.sin(a * x), 2)


class SEChannelAttention(nn.Module):
    """Recalibrates EnCodec latent channels to suppress static bias and amplify texture-specific channels."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        w = x.mean(dim=-1)  # [B, C]
        w = self.fc(w).unsqueeze(-1)  # [B, C, 1]
        return x * w


class DilatedResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.block = nn.Sequential(
            SnakeActivation(channels),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.GroupNorm(8, channels),
            SnakeActivation(channels),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.shortcut = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.block(x)


class AntiCheckerboardUpsampleBlock(nn.Module):
    """Replaces strided ConvTranspose1d with Linear Interpolation + Conv1d to eliminate spectral notches."""
    def __init__(self, in_ch: int, out_ch: int, scale_factor: int, kernel_size: int = 7):
        super().__init__()
        self.scale_factor = scale_factor
        pad = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.ReflectionPad1d(pad),
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size),
            nn.GroupNorm(8, out_ch),
        )
        self.res1 = DilatedResBlock(out_ch, kernel_size=3, dilation=1)
        self.res2 = DilatedResBlock(out_ch, kernel_size=3, dilation=3)
        self.res3 = DilatedResBlock(out_ch, kernel_size=3, dilation=9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="linear", align_corners=False)
        x = self.conv(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return x


class DifferentiableTactileFilterbank(nn.Module):
    """
    DDSP-style multi-band stochastic texture synthesizer.
    Shapes broadband Gaussian noise across 32 log-spaced frequency bands (30-800 Hz)
    using learned material-specific spectral gains and time-varying envelopes.
    """
    def __init__(self, num_bands: int = 32, filter_len: int = 257, sr: int = 16000, f_min: float = 30.0, f_max: float = 800.0):
        super().__init__()
        self.num_bands = num_bands
        self.filter_len = filter_len
        # Precompute fixed bandpass FIR filters for 32 log-spaced center frequencies in [30, 800 Hz]
        freqs = torch.logspace(math.log10(f_min), math.log10(f_max), num_bands)
        t = torch.arange(-(filter_len // 2), filter_len // 2 + 1, dtype=torch.float32) / sr
        window = torch.hann_window(filter_len)

        filters = []
        for i in range(num_bands):
            fc = freqs[i].item()
            bw = max(fc * 0.18, 15.0)
            # Sinc bandpass filter centered at fc
            h = 2.0 * (bw / sr) * torch.sinc(bw * t) * torch.cos(2.0 * math.pi * fc * t)
            h = h * window
            h = h / (torch.norm(h, p=2) + 1e-8)
            filters.append(h)

        # Shape: [num_bands, 1, filter_len]
        self.register_buffer("fir_bank", torch.stack(filters, dim=0).unsqueeze(1))

    def forward(self, band_envelopes: torch.Tensor) -> torch.Tensor:
        """
        band_envelopes: [B, num_bands, L] non-negative time-frequency gain masks
        Returns: [B, 1, L] shaped stochastic tactile vibration
        """
        B, K, L = band_envelopes.shape
        # Generate white noise [B, K, L]
        noise = torch.randn(B, K, L, device=band_envelopes.device, dtype=band_envelopes.dtype)
        # Filter each band independently using depthwise 1D convolution
        pad = self.filter_len // 2
        noise_padded = F.pad(noise, (pad, pad), mode="reflect")
        band_noise = F.conv1d(noise_padded, self.fir_bank, groups=K)  # [B, K, L]
        # Normalize each band to unit variance before applying learned time-frequency envelopes
        band_noise = band_noise / (band_noise.std(dim=-1, keepdim=True) + 1e-6)
        shaped = (band_noise * band_envelopes).sum(dim=1, keepdim=True) / math.sqrt(K)
        return shaped


class SlidingSEANetDecoder(nn.Module):
    """
    Hybrid Deterministic + Stochastic SEANet Decoder for Continuous Sliding Texture Synthesis.
    Eliminates 3.33x temporal compression, 540Hz spectral centroid collapse, and ConvTranspose1d notches.
    """
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        target_length: int = 4800,
        num_tactile_bands: int = 32,
        upsample_rates: tuple = (5, 4, 4, 2),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.target_length = target_length
        self.num_tactile_bands = num_tactile_bands

        # 1. Latent Normalization & Channel Attention (prevents EnCodec static offset collapse)
        self.latent_norm = nn.InstanceNorm1d(latent_dim, affine=True)
        self.channel_attn = SEChannelAttention(latent_dim, reduction=4)
        self.global_stats_proj = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

        # 2. Temporal Sequence Modeling
        self.input_proj = nn.Conv1d(latent_dim, hidden_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.lstm_norm = nn.LayerNorm(hidden_dim)

        # 3. Anti-Checkerboard Upsampling Backbone (30 frames -> 4800 samples via 5*4*4*2 = 160x)
        self.upsample_blocks = nn.ModuleList()
        in_ch = hidden_dim
        for rate in upsample_rates:
            out_ch = max(in_ch // 2, 32)
            self.upsample_blocks.append(AntiCheckerboardUpsampleBlock(in_ch, out_ch, scale_factor=rate))
            in_ch = out_ch

        self.final_ch = in_ch

        # 4A. Deterministic Waveform Head (Macro-grooves & transients)
        self.det_head = nn.Sequential(
            SnakeActivation(self.final_ch),
            nn.Conv1d(self.final_ch, 1, kernel_size=7, padding=3),
        )

        # 4B. Stochastic Filterbank Head (Micro-asperity broadband texture)
        self.band_env_head = nn.Sequential(
            SnakeActivation(self.final_ch),
            nn.Conv1d(self.final_ch, num_tactile_bands, kernel_size=5, padding=2),
            nn.Softplus(),
        )
        self.filterbank = DifferentiableTactileFilterbank(num_bands=num_tactile_bands)

        # 5. Global Mix & Gain Calibration
        self.mix_gate = nn.Sequential(
            nn.Linear(latent_dim * 2, 32),
            nn.GELU(),
            nn.Linear(32, 2),
            nn.Softplus(),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        latents: [B, 128, T] from EnCodec.
        If T == 75 (from 1.0s audio), slices the first 23 frames corresponding to the
        aligned 0.3s audio segment so time is 1:1 (no 3.33x compression).
        """
        if latents.size(-1) > 30:
            # Slice the aligned 0.3s portion (23 frames @ 75Hz frame rate)
            latents = latents[:, :, :23]

        # Extract global mean & std BEFORE InstanceNorm to retain material roughness/loudness cues
        feat_mean = latents.mean(dim=-1)
        feat_std = latents.std(dim=-1) + 1e-6
        global_stats = torch.cat([feat_mean, feat_std], dim=-1)  # [B, 256]

        # Normalize per-instance channel offsets and apply channel attention
        x = self.latent_norm(latents)
        x = self.channel_attn(x)

        # Interpolate latent frames cleanly to 30 frames (so 30 * 160 = 4800 samples)
        x = F.interpolate(x, size=30, mode="linear", align_corners=False)

        x = self.input_proj(x)  # [B, hidden_dim, 30]

        # FiLM conditioning from global latent statistics
        gamma_beta = self.global_stats_proj(global_stats).unsqueeze(-1)  # [B, 2*hidden_dim, 1]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        x = x * (1.0 + 0.2 * torch.tanh(gamma)) + 0.2 * beta

        # BiLSTM temporal modeling
        x_seq = x.transpose(1, 2)  # [B, 30, hidden_dim]
        lstm_out, _ = self.lstm(x_seq)
        x = x + self.lstm_norm(lstm_out).transpose(1, 2)

        # Upsample 30 -> 150 -> 600 -> 2400 -> 4800
        for block in self.upsample_blocks:
            x = block(x)

        if x.size(-1) != self.target_length:
            x = F.interpolate(x, size=self.target_length, mode="linear", align_corners=False)

        # Synthesize deterministic + stochastic components
        y_det = self.det_head(x)  # [B, 1, 4800]
        band_envs = self.band_env_head(x)  # [B, 32, 4800]
        y_stoch = self.filterbank(band_envs)  # [B, 1, 4800]

        # Material-adaptive mixing weights
        weights = self.mix_gate(global_stats).unsqueeze(-1)  # [B, 2, 1]
        w_det = weights[:, 0:1, :]
        w_stoch = weights[:, 1:2, :]

        out = w_det * y_det + w_stoch * y_stoch
        out = out - out.mean(dim=-1, keepdim=True)
        return F.hardtanh(out, min_val=-6.0, max_val=6.0)

```

#### STEP 4: Overhaul the Loss Function (`src/models/loss_sliding.py`)

##### Why the Current Loss Fails:

Pointwise time-domain MSE and L1 losses punish any phase-shifted oscillation, driving `peak_pred` down to $\sim 1.4$ and forcing the network into a degenerate $540\text{ Hz}$ carrier wave.

##### New Phase-Invariant Texture Composite Loss (`SlidingTextureLoss`):

1. **Zero Out Pointwise Time-Domain Waveform Loss (**$w_{\text{time}} = 0.0$**):** Completely remove sample-by-sample MSE/L1 between `pred` and `target`.

2. **Tactile-Band Welch/Multi-Window PSD Loss (**$\mathcal{L}_{\text{psd}}$**):**

   * Computes the Power Spectral Density within the $30\text{--}800\text{ Hz}$ tactile band and minimizes **both**:

     * **Passband Log-Spectral Distance (LSD in dB):** Directly minimizes the evaluation metric `lsd_tactile_band`.

     * **Passband Linear & Log PSD Cosine Distance:** Directly maximizes `psd_cosine_similarity`.

3. **Differentiable Spectral Centroid & Spread Loss (**$\mathcal{L}_{\text{moments}}$**):**

   * Computes the exact differentiable spectral centroid
     

     $$
     C(P) = \frac{\sum f \cdot P(f)}{\sum P(f)}
     $$

     
     and spectral spread
     

     $$
     \Delta f(P) = \sqrt{\frac{\sum (f - C)^2 P(f)}{\sum P(f)}}
     $$

     
     inside $[30, 800\text{ Hz}]$ and penalizes L1 error against the ground truth. This **forces the predicted centroid to track each material's true centroid across** $130\text{--}900\text{ Hz}$.

4. **Multi-Resolution STFT Log-Magnitude + Spectral Convergence Loss (**$\mathcal{L}_{\text{stft}}$**):**

   * Uses FFT sizes $(256, 512, 1024, 2048)$ restricted to tactile frequencies ($20\text{--}1000\text{ Hz}$).

5. **Multi-Scale RMS Envelope & Crest-Factor / Quantile Loss (**$\mathcal{L}_{\text{env\_stat}}$**):**

   * Matches multi-scale temporal RMS envelopes ($\text{win} \in \{64, 256, 512, 1024\}$) via L1 + Pearson correlation loss.

   * Matches global **RMS**, **99th-percentile Peak Amplitude (`torch.quantile(|x|, 0.99)`)**, and **Crest Factor (**$\text{Peak}/\text{RMS}$**)** so `peak_amplitude_ratio` and `rms_ratio` converge to $1.0\times$.

6. **Batch Spectral Diversity Loss (**$\mathcal{L}_{\text{div}}$**):**

   * Penalizes mode collapse by ensuring pairwise spectral distances between samples $(i, j)$ in a predicted batch match the pairwise spectral distances in the ground-truth batch.

##### Complete Drop-In Code for `src/models/loss_sliding.py`:

```
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingTextureLoss(nn.Module):
    """
    Phase-Invariant Perceptual & Spectral Texture Loss for Controlled Sliding Synthesis.
    Eliminates pointwise time-domain MSE/L1 (which causes phase cancellation and 540Hz collapse)
    and directly optimizes Tactile PSD shape, Spectral Centroid/Spread, Multi-Scale Envelopes,
    Crest Factor, and Batch Spectral Diversity.
    """
    def __init__(
        self,
        sr: int = 16000,
        f_min: float = 30.0,
        f_max: float = 800.0,
        w_stft: float = 2.0,
        w_psd: float = 4.0,
        w_centroid: float = 3.0,
        w_env: float = 2.5,
        w_stat: float = 2.0,
        w_div: float = 1.5,
    ):
        super().__init__()
        self.sr = sr
        self.f_min = f_min
        self.f_max = f_max
        self.w_stft = w_stft
        self.w_psd = w_psd
        self.w_centroid = w_centroid
        self.w_env = w_env
        self.w_stat = w_stat
        self.w_div = w_div
        self.fft_sizes = (256, 512, 1024, 2048)
        self.env_windows = (64, 256, 512, 1024)

    def _compute_tactile_psd_and_moments(self, y: torch.Tensor, n_fft: int = 1024, hop: int = 256):
        # y: [B, L]
        win = torch.hann_window(n_fft, device=y.device, dtype=y.dtype)
        stft = torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True)
        power = stft.abs().pow(2).mean(dim=-1)  # Welch-like average PSD across frames: [B, F]

        freqs = torch.linspace(0, self.sr / 2.0, power.size(-1), device=y.device, dtype=y.dtype)
        mask = (freqs >= self.f_min) & (freqs <= self.f_max)

        p_tac = power[:, mask] + 1e-8  # [B, F_tac]
        f_tac = freqs[mask].unsqueeze(0)  # [1, F_tac]

        # Normalized probability distribution over tactile frequencies
        p_norm = p_tac / (p_tac.sum(dim=-1, keepdim=True) + 1e-8)

        # Differentiable Spectral Centroid (Hz) & Spectral Spread (Hz)
        centroid = (p_norm * f_tac).sum(dim=-1)  # [B]
        spread = torch.sqrt(((f_tac - centroid.unsqueeze(-1)).pow(2) * p_norm).sum(dim=-1) + 1e-6)  # [B]

        # Log-PSD in dB (clamped at -60 dB below peak to prevent stopband floor blowup)
        log_p = 10.0 * torch.log10(p_tac)
        max_db = log_p.max(dim=-1, keepdim=True)[0]
        log_p = torch.maximum(log_p, max_db - 60.0)

        return p_tac, log_p, p_norm, centroid, spread

    def _multi_res_stft_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = 0.0
        for n_fft in self.fft_sizes:
            hop = n_fft // 4
            win = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
            s_p = torch.stft(pred, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True).abs()
            s_t = torch.stft(target, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True).abs()

            freqs = torch.linspace(0, self.sr / 2.0, s_p.size(1), device=pred.device, dtype=pred.dtype)
            mask = (freqs >= 20.0) & (freqs <= 1000.0)
            s_p_band = s_p[:, mask, :] + 1e-6
            s_t_band = s_t[:, mask, :] + 1e-6

            sc_loss = torch.norm(s_t_band - s_p_band, p="fro") / (torch.norm(s_t_band, p="fro") + 1e-6)
            mag_loss = F.l1_loss(torch.log(s_p_band), torch.log(s_t_band))
            total = total + sc_loss + mag_loss
        return total / len(self.fft_sizes)

    def _envelope_and_stats_loss(self, pred: torch.Tensor, target: torch.Tensor):
        # pred, target: [B, 1, L]
        env_loss = 0.0
        for w in self.env_windows:
            stride = max(w // 2, 1)
            env_p = torch.sqrt(F.avg_pool1d(pred.pow(2), kernel_size=w, stride=stride) + 1e-8)
            env_t = torch.sqrt(F.avg_pool1d(target.pow(2), kernel_size=w, stride=stride) + 1e-8)

            l1_env = F.l1_loss(env_p, env_t)
            # Pearson correlation on envelopes
            vp = env_p - env_p.mean(dim=-1, keepdim=True)
            vt = env_t - env_t.mean(dim=-1, keepdim=True)
            corr = (vp * vt).sum(dim=-1) / (torch.sqrt((vp.pow(2).sum(dim=-1) + 1e-8) * (vt.pow(2).sum(dim=-1) + 1e-8)))
            env_loss = env_loss + l1_env + 0.5 * (1.0 - corr.mean())
        env_loss = env_loss / len(self.env_windows)

        # Global RMS, 99th-Percentile Peak, and Crest Factor matching
        rms_p = torch.sqrt(pred.pow(2).mean(dim=-1) + 1e-8)
        rms_t = torch.sqrt(target.pow(2).mean(dim=-1) + 1e-8)
        rms_loss = F.l1_loss(rms_p, rms_t)

        peak99_p = torch.quantile(pred.abs(), 0.99, dim=-1)
        peak99_t = torch.quantile(target.abs(), 0.99, dim=-1)
        peak_max_p = pred.abs().amax(dim=-1)
        peak_max_t = target.abs().amax(dim=-1)
        peak_loss = F.l1_loss(peak99_p, peak99_t) + 0.5 * F.l1_loss(peak_max_p, peak_max_t)

        crest_p = peak99_p / (rms_p + 1e-6)
        crest_t = peak99_t / (rms_t + 1e-6)
        crest_loss = F.l1_loss(crest_p, crest_t)

        stat_loss = rms_loss + peak_loss + 0.5 * crest_loss
        return env_loss, stat_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)
        if target.dim() == 2:
            target = target.unsqueeze(1)

        p_flat = pred.squeeze(1)
        t_flat = target.squeeze(1)

        # 1. Tactile PSD & Spectral Moments
        p_tac, log_p, p_norm, cent_p, spread_p = self._compute_tactile_psd_and_moments(p_flat)
        t_tac, log_t, t_norm, cent_t, spread_t = self._compute_tactile_psd_and_moments(t_flat)

        # LSD in tactile band (scaled by 1/10 so magnitude is ~0.3-0.5)
        lsd_loss = torch.sqrt(F.mse_loss(log_p, log_t) + 1e-6) / 10.0
        # PSD Cosine Similarity loss (both linear power and normalized distribution)
        cos_lin = 1.0 - F.cosine_similarity(p_tac, t_tac, dim=-1).mean()
        cos_log = 1.0 - F.cosine_similarity(log_p, log_t, dim=-1).mean()
        psd_loss = lsd_loss + cos_lin + 0.5 * cos_log

        # Spectral Centroid & Spread Loss (normalized by f_max = 800 Hz)
        centroid_loss = (
            F.l1_loss(cent_p / self.f_max, cent_t / self.f_max)
            + 0.5 * F.l1_loss(spread_p / self.f_max, spread_t / self.f_max)
        )

        # 2. Multi-Resolution STFT Loss
        stft_loss = self._multi_res_stft_loss(p_flat, t_flat)

        # 3. Envelope & Statistical Crest-Factor Loss
        env_loss, stat_loss = self._envelope_and_stats_loss(pred, target)

        # 4. Batch Spectral Diversity Loss (prevents mode collapse across batch items)
        if p_flat.size(0) > 1:
            dist_p = torch.cdist(p_norm, p_norm, p=1)
            dist_t = torch.cdist(t_norm, t_norm, p=1)
            div_loss = F.l1_loss(dist_p, dist_t) + F.l1_loss(
                cent_p.std() / self.f_max, cent_t.std() / self.f_max
            )
        else:
            div_loss = torch.tensor(0.0, device=pred.device)

        total_loss = (
            self.w_psd * psd_loss
            + self.w_centroid * centroid_loss
            + self.w_stft * stft_loss
            + self.w_env * env_loss
            + self.w_stat * stat_loss
            + self.w_div * div_loss
        )
        return total_loss

```

#### STEP 5: Fix Stopband LSD Blowup in Evaluation Metrics (`src/evaluation/metrics.py`)

##### Changes Required:

In `src/evaluation/metrics.py`, inspect how `log_spectral_distance` (full-band) is computed. Currently, because `accelDFT` is bandpass-filtered to $30\text{--}800\text{ Hz}$, bins in $800\text{--}8000\text{ Hz}$ contain near-zero filter stopband energy ($10^{-12}$ to $10^{-6}$), causing $10 \log_{10}(P)$ to produce $-120\text{ dB}$ vs. $-60\text{ dB}$ ($35+\text{ dB}$ artificial error).

* **Fix:** Clamp `P_gt` and `P_pred` dynamic range floor to $-60\text{ dB}$ ($10^{-6}$) relative to each signal's peak PSD bin before taking $\log_{10}$:

```
p_gt_floor = np.maximum(p_gt, np.max(p_gt) * 1e-6 + 1e-12)
p_pred_floor = np.maximum(p_pred, np.max(p_pred) * 1e-6 + 1e-12)

```

This keeps `lsd_tactile_band` accurate while preventing `log_spectral_distance` from blowing up to $38\text{ dB}$ on stopband numerical noise.

#### STEP 6: Training & Verification Pipeline (`scripts/train_sliding.py`, `scripts/evaluate_sliding.py`, `scripts/inference_sliding.py`)

1. In `scripts/train_sliding.py`:

   * Use `AdamW(decoder.parameters(), lr=3e-4, weight_decay=1e-4)` with `CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)`.

   * Clip gradients at `max_norm=2.0`.

   * Save `checkpoints/best_sliding_decoder.pt` based on best validation `SlidingTextureLoss`.

2. Execute the full 4-step pipeline and verify:

   * `uv run python scripts/run_preprocessing_sliding.py` (or `run_preprocessing.py`)

   * `uv run python scripts/train_sliding.py` (or `train.py`)

   * `uv run python scripts/evaluate_sliding.py` (or `evaluate.py`)

   * `uv run python scripts/inference_sliding.py` (or `inference.py`)

3. **Success Criteria to Verify in `evaluation_report.json`:**

   * No `"tmpC1"` / `"Unknown"` class entries in `per_sample_results`.

   * `spectral_centroid_pred_hz` varies dynamically across $150\text{--}750\text{ Hz}$ matching `spectral_centroid_gt_hz` (`spectral_centroid_delta_hz` drops from $115.8\text{ Hz}$ to $< 55\text{ Hz}$).

   * `peak_amplitude_ratio` improves from $0.468$ to $0.85\text{--}1.10$, and `rms_ratio` improves from $0.697$ to $0.90\text{--}1.05$.

   * `lsd_tactile_band` improves from $4.94\text{ dB}$ to $< 2.8\text{ dB}$, and `psd_cosine_similarity` improves from $0.546$ to $> 0.80$.

   * Update `context.md` with the final post-training metrics.

## Part 3: Updated `context.md`

# Project Context — Audio-to-Haptic Cross-Modal Synthesis

> **Last Updated:** 2026-11-01 17:05 IST
>
> **Project:** BTP — Understanding Multisensory Information to Build Perception of Object Properties
>
> **Author:** ParthKhiriya (b23ee1051@iitj.ac.in)
>
> **Current Task:** Phase 2.2 — Controlled Sliding Synthesis Overhaul (Resolving Spectral Mode Collapse & Temporal Compression)

### 1. Project Overview

This BTP investigates cross-modal relationships between vision, sound, and haptics during interactions with textured surfaces. The immediate goal is building a generative model that synthesizes perceptually meaningful haptic (vibrotactile) signals directly from audio input.

#### Research Objectives

1. **Cross-Modal Learning** — Predict haptic signals from audio (and later visual) modalities

2. **Multimodal Interaction** — Investigate relative weighting of modalities during texture interaction

3. **Physical Features** — Study perceptually validated features (hardness, roughness)

4. **Perceptual Dimensions** — Characterize perceptual space of textured surfaces during tool-tip interactions

### 2. Literature Review

| Paper | Architecture | Task | Key Details | 
| ----- | ----- | ----- | ----- | 
| **Sound2Hap** (Li & Seifi, CHI '26) | CNN autoencoder (EnCodec encoder + SEANet decoder) | Audio $\to$ vibrotactile | Frozen EnCodec encoder, SEANet decoder with LSTM + residual upsampling, LeakyReLU, no tanh output. Trained with human perceptual ratings. | 
| **HapticGen** (Sung et al., CHI '25) | Autoregressive transformer + EnCodec | Text $\to$ vibrotactile | Leverages EnCodec neural audio codec | 
| **VibNet** | Deep learning framework | Predict roughness/emotion from haptic | Biomimetic approach | 
| **Cluster Haptic Texture Dataset** (Eguchi, 2026) | — | Dataset: 118 surfaces | 3-axis machine, controlled velocity/direction | 
| **Devillard's Dataset** (Imperial, 2024-25) | — | Dataset: 10 surfaces | Bare-finger, multimodal | 
| **Haptipedia** (Agrawal et al.) | — | Device corpus | Taxonomy for haptic device design | 

### 3. Dataset — LMT Haptic Texture Database (TUM CBM)

#### 3.1 Taxonomy and File Naming Structure

Every `.mat` file in `data/LMT_FinalDatabase/CBM_FinalDatabase/` follows the standard format:
`C<class>_S<subclass>_M<material>_P<processing>_X<composition>_I<instance>.mat`

* **C (Class 1–8):**

  * C1: Wood (Softwood, Hardwood, Barks, Celluloids, Crafted Wood)

  * C2: Paper / Fabric / Biodegradable (Natural fibers, Fur, Leather)

  * C3: Polymers / Plastics (Thermoplasts, Thermosets, Elastomers, Carbon fibers)

  * C4: Ceramics (Earthenware, Stoneware, High-tech ceramics)

  * C5: Glasses (Crafted glasses, Natural glasses)

  * C6: Stones (Igneous, Sedimentary, Metamorphic, Cement, Minerals)

  * C7: Metals (Ferrous, Non-ferrous, Alloys, Metal fibers)

  * C8: Composites (Polymer-Stone, Polymer-Metal, Wood-Stone)

* **S (Subclass):** Geological, biological, or chemical category.

* **M (Material ID):** Specific material specimen (e.g., spruce, pine, granite, steel).

* **P (Processing / Shape):**

  * Wood: P1 veneered tangential, P2 cross-section, P3 sliced, P4 natural tangential, P5 natural cross-section.

  * Stone: P1 stacked, P2 polished, P3 pointed, P4 sanded, P5 embossed.

  * Metals: P1 steamrolled, P2 polished, P3 casted, P4 perforated.

* **X (Composition / Moisture / Finish):**

  * Wood: moisture percentage.

  * Polymers/Metals: purity percentage.

  * Glass: opacity level.

  * Composites/Abrasives: grit size (X4=P40, X8=P80, X24=P240, X60=P600).

* **I (Instance 1–5):** 5 repeated independent measurement passes per surface.

Total: 1,115 valid material files across all 8 classes.

#### 3.2 Modality & Interaction Decomposition

Each `.mat` file bundles multiple distinct physical interactions and sensing modalities:

1. **Controlled Tap Interaction (Impact Contact)**:

   * `soundTap`: $5.0\text{ s}$ @ $44,100\text{ Hz}$ ($220,500$ samples) contact acoustic tap.

   * `accelTap`: $1.5\text{ s}$ @ $10,000\text{ Hz}$ ($15,003$ samples) impact acceleration.

   * `normalForce`: $1.5\text{ s}$ @ $10,000\text{ Hz}$ ($15,003$ samples) dynamic normal contact force.

   * `frictionForce`: $1.5\text{ s}$ @ $10,000\text{ Hz}$ ($15,003$ samples) dynamic lateral force.

   * `soundTapSteelTooltip` & `accelTapSteelTooltip`: tool-tip variant using customized stainless-steel tip.

2. **Continuous Sliding / Stroking Interaction (Surface Texture Friction)**:

   * `sound` / `soundSteelTooltip`: $1.0\text{ s}$ @ $44,100\text{ Hz}$ ($44,100$ samples) sliding friction sound.

   * `accelDFT` / `accelDFTSteelTooltip`: $0.3\text{ s}$ @ $10,000\text{ Hz}$ ($3,000$ samples) sliding vibration, reduced from 3D to 1D via Kuchenbecker's `DFT321` algorithm.

3. **Mechanical Compliance / Flexibility**:

   * `pressure`, `pressureFolding`, `foldingAngle`: $12,500$ samples measuring bending resistance.

4. **Thermal, Electrical & Optical Sensors**:

   * `temperatureData`: $2,240$ samples measuring heat dissipation into the material.

5. **Visual Photography**:

   * `displayImage`, `macroImage`, `illuminatedImage`: high-resolution $(4032 \times 3024 \times 3)$ RGB photography.

#### 3.3 Disambiguation: Controlled vs. Uncontrolled vs. Tap vs. Sliding

| Dimension | Concept 1 | Concept 2 | 
| ----- | ----- | ----- | 
| **Actuator / Operator** | **Controlled (Robotic / Machine):** Performed by the **Texplorer / Phantom Omni** robotic arm. Velocity, force, and stylus orientation are electronically controlled. | **Uncontrolled / Unconstrained (Human Hand):** Performed by a human operator holding the stylus naturally in freehand movement. | 
| **Physical Motion** | **Tap (Impact):** Perpendicular vertical impact to study contact hardness, elasticity, and impact transients (`soundTap`, `accelTap`). | **Continuous Sliding (Friction):** Lateral dragging across the surface to study texture ridges, roughness, and friction (`sound`, `accelDFT`). | 

### 4. Phase 1: Controlled Tap Synthesis (`controlled-tap` Branch — Finalized Baseline)

#### 4.1 Pipeline & Quantitative Summary ($N = 186$ Validation Samples)

* **Envelope Correlation:** `Env Corr (64)`: $0.9275$, `Env Corr (256)`: $0.9577$, `Env Corr (1024)`: $0.9705$

* **Amplitude & Energy:** `Peak Ratio`: $1.1688$, `RMS Ratio`: $0.8264$, `Tactile Band Energy Ratio (50–300 Hz)`: $0.8496$

* **Spectral Fidelity:** `Tactile Band LSD`: $3.45\text{ dB}$ (median $2.48\text{ dB}$), `PSD CosSim`: $0.8100$, `Centroid Delta`: $36.95\text{ Hz}$

* **Time Domain:** `PCC`: $0.2327$, `RMSE`: $0.5680$, `SNR`: $-0.52\text{ dB}$

### 5. Phase 2: Controlled Sliding Synthesis (`controlled-sliding` Branch)

#### 5.1 Phase 2.1 Baseline Evaluation Results ($N = 201$ Validation Samples)

Initial evaluation of `best_sliding_decoder.pt` (`audio_length=16000`, `haptic_length=4800`, tactile band $30\text{--}800\text{ Hz}$) revealed severe degradation compared to Phase 1:

* **Time Domain & Amplitude:**

  * `Pearson Correlation (PCC)`: $0.0004 \pm 0.0646$ (median $0.0012$)

  * `RMSE`: $1.1874 \pm 0.0671$

  * `SNR (dB)`: $-1.7133 \pm 0.3668\text{ dB}$

  * `Peak Amplitude Ratio`: $0.4685 \pm 0.1466$ (median $0.4425$)

  * `RMS Energy Ratio`: $0.6968 \pm 0.0692$ (median $0.6833$)

* **Envelope Correlation:**

  * `Envelope Corr (64)`: $0.3873 \pm 0.2605$

  * `Envelope Corr (256)`: $0.5298 \pm 0.3226$

  * `Envelope Corr (1024)`: $0.6205 \pm 0.3907$

* **Spectral Fidelity:**

  * `Tactile Band (30–800 Hz) LSD`: $4.9373 \pm 0.8465\text{ dB}$ (median $4.79\text{ dB}$)

  * `Full-Band LSD`: $12.4792 \pm 13.2545\text{ dB}$ (stopband floor blowup)

  * `PSD Cosine Similarity`: $0.5456 \pm 0.1345$

  * `Spectral Centroid Delta`: $115.78 \pm 89.95\text{ Hz}$

  * `SPQI-Lite`: $6.1075 \pm 1.1457$

#### 5.2 Forensic Root-Cause Diagnosis of Phase 2.1 Failure

1. **Spectral Mode Collapse (**$\sim 540\text{ Hz}$ **Centroid Lock):** Every validation prediction across all 8 classes (`C1`–`C8`) exhibited a locked spectral centroid between $535\text{ Hz}$ and $552\text{ Hz}$ and constant spectral flatness ($\sim 2.2 \times 10^{-6}$), ignoring input material variations (`GT Centroid` spanning $138\text{ Hz}$ to $1006\text{ Hz}$).

2. **Periodic Sinusoidal & Comb-Filter Artifacts:** Strided `ConvTranspose1d` layers upsampling static EnCodec latents produced a deterministic $\sim 540\text{ Hz}$ carrier wave amplitude-modulated at $75\text{ Hz}$ (the EnCodec frame rate) with deep comb notches above $700\text{ Hz}$, failing to reproduce stochastic frictional micro-impacts (`Crest Factor` $\sim 2.0$ vs. GT $\sim 4.5$).

3. **Pointwise Time-Domain Loss Conflict:** Because `sound` and `accelDFT` are stochastic and not phase-locked at the sub-millisecond level, pointwise MSE/L1 loss penalized valid high-frequency texture oscillations, suppressing peak amplitudes ($0.468\times$) and driving `SNR` to $-1.71\text{ dB}$.

4. $3.33\times$ **Temporal Compression & Filter Boundary Transients:** Compressing $1.0\text{ s}$ of audio ($16,000$ samples) into $0.3\text{ s}$ of haptics ($4,800$ samples) warped temporal envelopes by $3.33\times$. Additionally, FFT resampling (`scipy.signal.resample`) on raw `accelDFT` created large boundary spikes at $t = 0.00\text{ s}$ and $t = 0.30\text{ s}$ that dominated peak normalization (`Peak = 5.00`) and caused the decoder to learn a flat $0.6$ RMS line with edge flares.

5. **Corrupt Temp Files & Stopband LSD Instability:** Unfiltered `tmpC1_*.mat` files leaked into the dataset as class `"Unknown"`, and unclamped stopband power ($800\text{--}8000\text{ Hz}$) inflated full-band LSD to $35\text{--}40\text{ dB}$.

#### 5.3 Phase 2.2 Overhaul Architecture (Current Implementation Plan)

1. **Preprocessing (`src/data/preprocess_sliding.py`):**

   * Strict regex filtering (`^C[1-8]_S\d+_M\d+_P\d+_X\d+_I\d+$`) to purge `tmp*` files.

   * Pre-filter $6\%$ Tukey tapering + polyphase resampling (`resample_poly`) + odd-padded zero-phase SOS bandpass filtering ($30\text{--}800\text{ Hz}$) to eliminate $t = 0.0\text{ s} / 0.3\text{ s}$ edge spikes.

   * 1:1 envelope cross-correlation alignment extracting the best-matched $0.3\text{ s}$ ($4,800$-sample) audio segment so audio and haptic streams share an exact $1.0\times$ timescale.

   * Robust unit-RMS scaling with soft `tanh` outlier clipping at $[-5.0, 5.0]$.

2. **Hybrid Stochastic-Deterministic Decoder (`src/models/decoder_sliding.py`):**

   * `InstanceNorm1d` + Squeeze-and-Excitation (`SEChannelAttention`) + global `FiLM` conditioning on EnCodec latents ($T = 23$ frames for $0.3\text{ s}$).

   * Anti-checkerboard upsampling (`F.interpolate` + `ReflectionPad1d` + `Conv1d` + `SnakeActivation` dilated residual blocks).

   * Dual-stream synthesis combining a deterministic macro-groove head with a **DDSP-inspired 32-band Differentiable Tactile Filterbank** ($30\text{--}800\text{ Hz}$) shaping Gaussian noise to restore natural crest factor ($3.5\text{--}5.0$) and material-specific PSD shapes.

3. **Phase-Invariant Texture Loss (`src/models/loss_sliding.py`):**

   * Removed pointwise time-domain MSE/L1 ($w_{\text{time}} = 0.0$).

   * Added direct **Tactile-Band Welch PSD Loss** (LSD + linear/log Cosine Distance), **Differentiable Spectral Centroid & Spread Loss**, **Multi-Resolution STFT Loss**, **Multi-Scale RMS + 99th-Percentile Crest-Factor Loss**, and **Batch Spectral Diversity Loss**.

### 6. Execution & Implementation Status

* \[x\] Controlled Tap Synthesis Phase (`controlled-tap`): Validated & Finalized

* \[x\] Phase 2.1 Initial Sliding Pipeline Implemented & Evaluated ($N = 201$ samples)

* \[x\] Phase 2.1 Forensic Diagnosis Completed (Identified $540\text{ Hz}$ centroid lock, $3.33\times$ compression, edge spikes, and MSE phase cancellation)

* \[ \] Phase 2.2 Overhaul Execution:

  * \[ \] Apply `preprocess_sliding.py` overhaul and re-generate `data/processed/`

  * \[ \] Apply `SlidingSEANetDecoder` overhaul in `src/models/decoder_sliding.py`

  * \[ \] Apply `SlidingTextureLoss` in `src/models/loss_sliding.py`

  * \[ \] Fix stopband LSD floor clamping in `src/evaluation/metrics.py`

  * \[ \] Retrain sliding model (`scripts/train_sliding.py`) and re-run evaluation (`scripts/evaluate_sliding.py`)
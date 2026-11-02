# Project Context — Audio-to-Haptic Cross-Modal Synthesis

> **Last Updated:** 2026-11-02 22:30 IST  
> **Project:** BTP — Understanding Multisensory Information to Build Perception of Object Properties  
> **Author:** ParthKhiriya (b23ee1051@iitj.ac.in)  
> **Current Task:** Phase 2.4 — Publication-Grade Controlled Sliding Overhaul

---

## 1. Project Overview

This BTP investigates cross-modal relationships between vision, sound, and haptics during interactions with textured surfaces. The immediate goal is building a generative model that synthesizes perceptually meaningful haptic (vibrotactile) signals directly from audio input.

### Research Objectives
1. **Cross-Modal Learning** — Predict haptic signals from audio (and later visual) modalities
2. **Multimodal Interaction** — Investigate relative weighting of modalities during texture interaction
3. **Physical Features** — Study perceptually validated features (hardness, roughness)
4. **Perceptual Dimensions** — Characterize perceptual space of textured surfaces during tool-tip interactions

---

## 2. Literature Review

| Paper | Architecture | Task | Key Details |
|---|---|---|---|
| **Sound2Hap** (Li & Seifi, CHI '26) | CNN autoencoder (EnCodec encoder + SEANet decoder) | Audio $\to$ vibrotactile | Frozen EnCodec encoder, SEANet decoder with LSTM + residual upsampling, LeakyReLU, no tanh output. Trained with human perceptual ratings. |
| **HapticGen** (Sung et al., CHI '25) | Autoregressive transformer + EnCodec | Text $\to$ vibrotactile | Leverages EnCodec neural audio codec |
| **VibNet** | Deep learning framework | Predict roughness/emotion from haptic | Biomimetic approach |
| **Cluster Haptic Texture Dataset** (Eguchi, 2026) | — | Dataset: 118 surfaces | 3-axis machine, controlled velocity/direction |
| **Devillard's Dataset** (Imperial, 2024-25) | — | Dataset: 10 surfaces | Bare-finger, multimodal |
| **Haptipedia** (Agrawal et al.) | — | Device corpus | Taxonomy for haptic device design |

---

## 3. Dataset — LMT Haptic Texture Database (TUM CBM)

### 3.1 Taxonomy and File Naming Structure
Every `.mat` file in `data/LMT_FinalDatabase/CBM_FinalDatabase/` follows the standard format:  
`C<class>_S<subclass>_M<material>_P<processing>_X<composition>_I<instance>.mat`

- **C (Class 1–8):** C1 Wood, C2 Paper/Fabric, C3 Polymers/Plastics, C4 Ceramics/Foam, C5 Glasses/Carpet/Leather, C6 Stones, C7 Metals, C8 Composites.
- **I (Instance 1–5):** 5 repeated independent measurement passes per surface.
- **Total:** 1,115 raw files; 1,000 valid paired recordings (800 train / 200 stratified validation).

### 3.2 Modality & Interaction Decomposition
1. **Controlled Tap Interaction (Impact Contact):** `soundTap` (5.0s @ 44.1 kHz) $\to$ `accelTap` (1.5s @ 10 kHz).
2. **Continuous Controlled Sliding (Surface Texture Friction):** `sound` / `soundSteelTooltip` (1.0s @ 44.1 kHz) $\to$ `accelDFT` / `accelDFTSteelTooltip` (0.3s @ 10 kHz, 1D via `DFT321`).

---

## 4. Phase 1: Controlled Tap Synthesis (`controlled-tap` Branch — Finalized Baseline)

### 4.1 Quantitative Summary ($N = 186$ Validation Samples)
- **Envelope Correlation:** `Env Corr (64)`: **0.9275**, `Env Corr (256)`: **0.9577**, `Env Corr (1024)`: **0.9705**
- **Amplitude & Energy:** `Peak Ratio`: **1.1688**, `RMS Ratio`: **0.8264**, `Tactile Band Energy Ratio (50–300 Hz)`: **0.8496**
- **Spectral Fidelity:** `Tactile Band LSD`: **3.45 dB** (median **2.48 dB**), `PSD CosSim`: **0.8100**, `Centroid Delta`: **36.95 Hz**
- **Time Domain:** `PCC`: **0.2327**, `RMSE`: **0.5680**, `SNR`: **-0.52 dB**

---

## 5. Phase 2: Controlled Sliding Synthesis (`controlled-sliding` Branch)

### 5.1 Progression Across Overhaul Phases ($N = 200$ Stratified Validation Samples)

| Metric | Phase 2.1 (Baseline) | Phase 2.2 (Stochastic) | Phase 2.3 (Acoustic-Guided) | **Phase 2.4 (Publication-Grade)** | Ideal |
|---|---|---|---|---|---|
| **Tactile Band (30–800Hz) LSD** | $4.94 \pm 0.85\text{ dB}$ | $3.46 \pm 1.06\text{ dB}$ | $3.17 \pm 0.83\text{ dB}$ (med $3.05\text{ dB}$) | **$3.07 \pm 0.92\text{ dB}$** (med **$2.83\text{ dB}$**) | $\to 0.0$ |
| **Full-Band LSD (0–8 kHz)** | $12.48 \pm 13.25\text{ dB}$ | $5.58 \pm 1.65\text{ dB}$ | $3.59 \pm 1.77\text{ dB}$ (med $3.16\text{ dB}$) | **$5.30 \pm 1.94\text{ dB}$** (med **$4.86\text{ dB}$**) | $\to 0.0$ |
| **SPQI-Lite (Weighted LSD)** | $6.11 \pm 1.15$ | $5.87 \pm 1.04$ | $3.54 \pm 0.75$ (med $3.44$) | **$5.68 \pm 0.90$** (med **$5.63$**) | $\to 0.0$ |
| **PSD Cosine Similarity** | $0.546 \pm 0.135$ | $0.653 \pm 0.127$ | $0.735 \pm 0.127$ (med $0.766$) | **$0.757 \pm 0.129$** (med **$0.785$**) | $\to 1.0$ |
| **Spectral Convergence** | $0.839 \pm 0.070$ | $0.782 \pm 0.152$ | $0.680 \pm 0.144$ (med $0.663$) | **$0.683 \pm 0.186$** (med **$0.672$**) | $\to 0.0$ |
| **Spectral Centroid Delta** | $115.78 \pm 89.95\text{ Hz}$ | $96.06 \pm 62.60\text{ Hz}$ | $65.96 \pm 55.15\text{ Hz}$ (med $50.61\text{ Hz}$) | **$71.67 \pm 58.03\text{ Hz}$** (med **$60.74\text{ Hz}$**) | $\to 0.0$ |
| **RMS Energy Ratio** | $0.697 \pm 0.069$ | $0.997 \pm 0.098$ | $0.973 \pm 0.016$ (med $0.973$) | **$0.990 \pm 0.015$** (med **$0.987$**) | $\to 1.0$ |
| **Tactile Band Energy Ratio** | $0.663 \pm 1.029$ | $0.841 \pm 0.201$ | $0.909 \pm 0.086$ (med $0.904$) | **$1.048 \pm 0.085$** (med **$1.039$**) | $\to 1.0$ |
| **Peak Amplitude Ratio** | $0.468 \pm 0.147$ | $1.005 \pm 0.192$ | $1.100 \pm 0.198$ (med $1.070$) | **$1.036 \pm 0.169$** (med **$1.011$**) | $\to 1.0$ |
| **Envelope Corr (64 / 4ms)** | $0.387 \pm 0.261$ | $0.277 \pm 0.138$ | $0.365 \pm 0.133$ (med $0.350$) | **$0.402 \pm 0.151$** (med **$0.385$**) | $\to 1.0$ |
| **Envelope Corr (256 / 16ms)** | $0.530 \pm 0.323$ | $0.473 \pm 0.190$ | $0.533 \pm 0.192$ (med $0.543$) | **$0.504 \pm 0.216$** (med **$0.510$**) | $\to 1.0$ |
| **Envelope Corr (1024 / 64ms)** | $0.620 \pm 0.391$ | $0.617 \pm 0.288$ | $0.634 \pm 0.295$ (med $0.710$) | **$0.635 \pm 0.288$** (med **$0.713$**) | $\to 1.0$ |
| **Pearson Correlation (PCC)** | $+0.0004 \pm 0.0646$ | $-0.0038 \pm 0.0394$ | $+0.0386 \pm 0.0707$ (max $+0.503$) | **$+0.0668 \pm 0.0879$** (med **$+0.053$**, max **$+0.503$**) | $\to 1.0$ |
| **RMSE / SNR (dB)** | $1.19$ / $-1.71\text{ dB}$ | $1.37$ / $-3.01\text{ dB}$ | $1.36 \pm 0.06$ / $-2.71 \pm 0.36\text{ dB}$ | **$1.35 \pm 0.07$ / $-2.64 \pm 0.44\text{ dB}$** | — |

### 5.2 Phase 2.4 Per-Class Performance Breakdown ($N = 200$)

| Class | Count | PCC | RMSE | SNR (dB) | Peak Ratio | Tactile LSD (dB) | PSD CosSim | SPQI-Lite |
|---|---|---|---|---|---|---|---|---|
| **C1 (Wood)** | 88 | +0.053 | 1.365 | -2.72 | 1.012 | 3.13 | 0.746 | 5.69 |
| **C2 (Paper/Fabric)** | 24 | **+0.110** | **1.314** | **-2.38** | 1.118 | 3.27 | 0.726 | 5.65 |
| **C3 (Plastic/Rubber)** | 28 | +0.086 | 1.343 | -2.55 | 1.093 | 3.20 | 0.750 | 5.48 |
| **C4 (Foam)** | 7 | +0.062 | 1.364 | -2.70 | 0.987 | 3.04 | 0.772 | 6.09 |
| **C5 (Carpet/Leather)** | 4 | +0.075 | 1.352 | -2.60 | 1.058 | **2.44** | **0.846** | **4.85** |
| **C6 (Stone/Ceramic)** | 31 | +0.056 | 1.366 | -2.71 | 0.976 | **2.77** | 0.782 | 5.77 |
| **C7 (Metal)** | 12 | +0.081 | 1.341 | -2.57 | 1.076 | 3.18 | 0.782 | 5.76 |
| **C8 (Glass)** | 6 | +0.040 | 1.374 | -2.78 | 1.054 | **2.50** | **0.825** | 6.01 |

### 5.3 Phase 2.4 Publication-Grade Overhaul Architecture
1. **Band-Matched Vectorized 1-Sample FFT Alignment & Phase/Polarity Locking (`preprocess_sliding.py`):**
   - Filters 1.0s audio to `30–800 Hz` and `30–350 Hz` (removing high-frequency air hiss).
   - Simultaneously correlates all 1-sample offsets in O(N log N) via FFT across 64, 256, 1024 RMS envelopes and waveform phase.
   - Synchronizes accelerometer polarity directly in the preprocessed files.
2. **Acoustic-Guided Hybrid Coherent + Dual-Resolution Decoder (`decoder_sliding.py`):**
   - **Unbiased Spectral Tilt Conditioning:** Combines z-score peak contrast with unnormalized relative dB tilt and low-frequency energy ratios.
   - **40-Band Filterbank Strictly in `[28, 800 Hz]` with Global-Std Normalization:** Replaces local-RMS carrier whitening with global per-band variance normalization, preserving rapid 4–16ms friction bursts.
   - **Removal of +0.05 Floor & Learned Spectral Tilt Gate:** Allows low-centroid surfaces to drop high frequencies by >50 dB, eliminating the 380–460 Hz pile-up.
   - **DifferentiableFineSpectralShaper (50 Welch + 232 rFFT bins):** Sharpens narrow 15.625 Hz modal peaks.
   - **MultiScaleEnvelopeFusionTCN (16 channels):** Predicts band envelopes and post-filter master envelope with multi-scale skip connections.
   - **Direct Phase-Coherent Waveform Head:** Translates tactile sub-bands for positive PCC and high SNR.
   - **Soft-Knee Crest Limiter + 12th-Order Zero-Phase FFT Bandpass:** Removes out-of-band leakage and clips spikes before linear unit-RMS calibration.
3. **Exact Welch PSD + Outlier-Weighted Dual-Domain Loss (`loss_sliding.py`):**
   - Exact 1024-pt Welch PSD matching `scipy.signal.welch`.
   - 10x resonant modal peak-weighted PSD loss.
   - Outlier-weighted centroid loss (2.2x weight on `< 260 Hz`).
   - Stride-1 multi-scale envelope correlation loss.
   - Multi-band scale-invariant waveform PCC loss.

---

## 6. Execution & Implementation Status

- [x] Controlled Tap Synthesis Phase (`controlled-tap`): Validated & Finalized
- [x] Phase 2.1 Initial Sliding Pipeline Implemented & Evaluated ($N = 201$ samples)
- [x] Phase 2.2 Stochastic Filterbank Overhaul Implemented & Evaluated ($N = 200$ samples)
- [x] Phase 2.3 Acoustic-Guided Coherent Overhaul Implemented, Trained (150 Epochs), & Evaluated ($N = 200$ samples)
- [x] Phase 2.4 Architecture & Loss Overhaul Implemented & Evaluated:
  - [x] Vectorized 1-sample FFT alignment with phase/polarity locking (`preprocess_sliding.py`)
  - [x] Co-augmentation synchrony verified (`dataset_sliding.py`)
  - [x] Acoustic-guided hybrid coherent + dual-resolution decoder (`decoder_sliding.py`)
  - [x] Exact Welch PSD, outlier-weighted centroid, and stride-1 envelope loss (`loss_sliding.py`)
  - [x] Training (150 Epochs, Checkpoint at Epoch 146) and 200-sample validation benchmark suite verified (`scripts/*`)
  - [x] Benchmark validated: Tactile LSD 3.07 dB (med 2.83 dB), Peak Ratio 1.011x, RMS Ratio 0.990, PSD CosSim 0.785 med, PCC +0.067

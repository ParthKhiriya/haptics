# Project Context — Audio-to-Haptic Cross-Modal Synthesis

> **Last Updated:** 2026-11-01 22:50 IST  
> **Project:** BTP — Understanding Multisensory Information to Build Perception of Object Properties  
> **Author:** ParthKhiriya (b23ee1051@iitj.ac.in)  
> **Current Task:** Phase 2.3 Evaluation Assessment — Controlled Sliding Synthesis

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

| Metric | Phase 2.1 (Baseline) | Phase 2.2 (Stochastic) | **Phase 2.3 (Acoustic-Guided)** | Ideal |
|---|---|---|---|---|
| **Tactile Band (30–800Hz) LSD** | $4.94 \pm 0.85\text{ dB}$ | $3.46 \pm 1.06\text{ dB}$ | **$3.17 \pm 0.83\text{ dB}$** (med **$3.05\text{ dB}$**) | $\to 0.0$ |
| **Full-Band LSD (0–8 kHz)** | $12.48 \pm 13.25\text{ dB}$ | $5.58 \pm 1.65\text{ dB}$ | **$3.59 \pm 1.77\text{ dB}$** (med **$3.16\text{ dB}$**) | $\to 0.0$ |
| **SPQI-Lite (Weighted LSD)** | $6.11 \pm 1.15$ | $5.87 \pm 1.04$ | **$3.54 \pm 0.75$** (med **$3.44$**) | $\to 0.0$ |
| **PSD Cosine Similarity** | $0.546 \pm 0.135$ | $0.653 \pm 0.127$ | **$0.735 \pm 0.127$** (med **$0.766$**) | $\to 1.0$ |
| **Spectral Convergence** | $0.839 \pm 0.070$ | $0.782 \pm 0.152$ | **$0.680 \pm 0.144$** (med **$0.663$**) | $\to 0.0$ |
| **Spectral Centroid Delta** | $115.78 \pm 89.95\text{ Hz}$ | $96.06 \pm 62.60\text{ Hz}$ | **$65.96 \pm 55.15\text{ Hz}$** (med **$50.61\text{ Hz}$**) | $\to 0.0$ |
| **RMS Energy Ratio** | $0.697 \pm 0.069$ | $0.997 \pm 0.098$ | **$0.973 \pm 0.016$** (med **$0.973$**) | $\to 1.0$ |
| **Tactile Band Energy Ratio** | $0.663 \pm 1.029$ | $0.841 \pm 0.201$ | **$0.909 \pm 0.086$** (med **$0.904$**) | $\to 1.0$ |
| **Peak Amplitude Ratio** | $0.468 \pm 0.147$ | $1.005 \pm 0.192$ | **$1.100 \pm 0.198$** (med **$1.070$**) | $\to 1.0$ |
| **Envelope Corr (64 / 4ms)** | $0.387 \pm 0.261$ | $0.277 \pm 0.138$ | **$0.365 \pm 0.133$** (med **$0.350$**) | $\to 1.0$ |
| **Envelope Corr (256 / 16ms)** | $0.530 \pm 0.323$ | $0.473 \pm 0.190$ | **$0.533 \pm 0.192$** (med **$0.543$**) | $\to 1.0$ |
| **Envelope Corr (1024 / 64ms)** | $0.620 \pm 0.391$ | $0.617 \pm 0.288$ | **$0.634 \pm 0.295$** (med **$0.710$**) | $\to 1.0$ |
| **Pearson Correlation (PCC)** | $0.0004 \pm 0.0646$ | $-0.0038 \pm 0.0394$ | **$+0.0386 \pm 0.0707$** (max **$+0.503$**) | $\to 1.0$ |
| **RMSE / SNR (dB)** | $1.19$ / $-1.71\text{ dB}$ | $1.37$ / $-3.01\text{ dB}$ | **$1.36 \pm 0.06$** / **$-2.71 \pm 0.36\text{ dB}$** | — |

### 5.2 Phase 2.3 Per-Class Performance Breakdown ($N = 200$)

| Class | Count | PCC | RMSE | SNR (dB) | Peak Ratio | Tactile LSD (dB) | PSD CosSim | SPQI-Lite |
|---|---|---|---|---|---|---|---|---|
| **C1 (Wood)** | 88 | +0.042 | 1.362 | -2.71 | 1.086 | 3.23 | 0.718 | 3.59 |
| **C2 (Paper/Fabric)** | 24 | +0.036 | 1.355 | -2.67 | 1.153 | 3.40 | 0.707 | 3.64 |
| **C3 (Plastic/Rubber)** | 28 | +0.039 | 1.364 | -2.69 | 1.144 | 3.27 | 0.740 | 3.55 |
| **C4 (Foam)** | 7 | +0.018 | 1.380 | -2.79 | 1.036 | **2.99** | **0.802** | 3.63 |
| **C5 (Carpet/Leather)** | 4 | **+0.070** | **1.345** | **-2.57** | 1.144 | **2.48** | **0.829** | **2.58** |
| **C6 (Stone/Ceramic)** | 31 | +0.038 | 1.369 | -2.73 | 1.059 | **2.80** | **0.773** | **3.40** |
| **C7 (Metal)** | 12 | +0.022 | 1.374 | -2.79 | 1.098 | 3.52 | 0.726 | 3.71 |
| **C8 (Glass)** | 6 | +0.039 | 1.358 | -2.68 | 1.144 | **2.79** | **0.755** | **3.41** |

### 5.3 Phase 2.3 Diagnostic Summary (Strengths vs. Remaining Bottlenecks)
1. **Strengths Achieved in Phase 2.3:**
   - **Broke Training Plateau:** Trained cleanly through 150 epochs (best checkpoint at Epoch 147, `Val Loss: 8.4029`), eliminating the Epoch 34 early-stopping stall from Phase 2.2.
   - **Publication-Grade Spectral & Energy Fidelity:** `Tactile Band LSD` reached **3.17 dB** (surpassing the Phase 1 Controlled Tap baseline of 3.45 dB), `SPQI-Lite` improved by **42%** (`6.11` $\to$ `3.54`), `Full-Band LSD` dropped to **3.59 dB** via the 40-band Butterworth roll-off (`25–1600 Hz`), and `RMS Ratio` locked to **0.973 ± 0.016**.
   - **Dynamic Centroid Tracking:** Predicted centroids now span `179 Hz` to `686 Hz` across material classes with a median error of **50.61 Hz**.
2. **Remaining Bottlenecks for Optional Phase 2.4 Refinement:**
   - **Fine-Scale Envelope Tracking (`Env Corr 64 = 0.365`, `Env Corr 256 = 0.533`):** During training, `env` loss plateaued around `~0.64` because carrier local-RMS whitening and the shallow 2-layer `env_fusion` gate smoothed out rapid `4–16 ms` micro-asperity bursts.
   - **Low-Frequency Centroid Outliers:** A small subset of very low-centroid (`< 250 Hz`) surfaces are over-predicted around `~380–450 Hz` due to the `+0.05` additive floor in `band_gain_predictor` and z-score removal of spectral tilt in `_extract_acoustic_spectrum`.
   - **Hardtanh `6.0` Peak Clipping:** ~15% of validation samples touch `peak_pred = 6.0`, mildly inflating `Peak Ratio` (`1.10x`).

---

## 6. Execution & Implementation Status

- [x] Controlled Tap Synthesis Phase (`controlled-tap`): Validated & Finalized
- [x] Phase 2.1 Initial Sliding Pipeline Implemented & Evaluated ($N = 201$ samples)
- [x] Phase 2.2 Stochastic Filterbank Overhaul Implemented & Evaluated ($N = 200$ samples)
- [x] Phase 2.3 Acoustic-Guided Coherent Overhaul Implemented, Trained (150 Epochs), & Evaluated ($N = 200$ samples)
- [ ] Decide whether to finalize Phase 2.3 as the Controlled Sliding baseline or execute a targeted Phase 2.4 refinement for fine-scale envelope correlation (`Env Corr 64/256`) and low-centroid (`<250 Hz`) precision
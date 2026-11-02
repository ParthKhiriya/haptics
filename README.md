# Cross-Modal Audio-to-Tactile Synthesis for Surface Textures (BTP)
## Continuous Controlled Sliding Interaction (`controlled-sliding` Branch)

> **B.Tech Project:** Understanding Multisensory Information to Build Perception of Object Properties  
> **Author:** Parth Khiriya (b23ee1051@iitj.ac.in)  
> **Institution:** Indian Institute of Technology Jodhpur (IITJ)  
> **Status:** Phase 2.4 Publication-Grade Controlled Sliding Synthesis (Benchmark Validated)

---

## 📖 Overview

This branch is the dedicated, self-contained implementation for **Continuous Controlled Sliding Interaction** on textured surfaces from the **LMT Haptic Texture Database (TUM CBM)**.

During continuous sliding, friction vibrations between the tool-tip and micro-surface asperities produce complex, stochastic vibrotactile waveforms (`accelDFT`). The goal is synthesizing high-fidelity, perceptually aligned 1D vibrotactile acceleration waveforms directly from surface contact sliding audio (`sound`).

```
                    ┌───────────────────────────┐
                    │ Raw Contact Audio (1.0s)  │
                    │      16 kHz Sampling      │
                    └─────────────┬─────────────┘
                                  │
                   ┌──────────────┴──────────────┐
                   │                             │
                   ▼                             ▼
       ┌───────────────────────┐   ┌───────────────────────────┐
       │ Frozen EnCodec (1.5k) │   │ Differentiable STFT Bins  │
       │  Latent Sequences     │   │ & Unbiased Spectral Tilt  │
       └───────────┬───────────┘   └─────────────┬─────────────┘
                   │                             │
                   └──────────────┬──────────────┘
                                  ▼
                   ┌─────────────────────────────┐
                   │ Acoustic-Guided SEANet      │
                   │ FiLM + BiLSTM Sequence Core │
                   └──────────────┬──────────────┘
                                  ▼
                   ┌─────────────────────────────┐
                   │ 40-Band Butterworth         │
                   │ Coherent Filterbank         │
                   │ (28–800 Hz, Unit Basis)     │
                   └──────────────┬──────────────┘
                                  ▼
                   ┌─────────────────────────────┐
                   │ Differentiable Modal Shaper │
                   │  & MultiScale Envelope TCN  │
                   └──────────────┬──────────────┘
                                  ▼
                   ┌─────────────────────────────┐
                   │ Coherent Waveform Stream    │
                   │ + Soft-Knee Crest Limiter   │
                   └──────────────┬──────────────┘
                                  ▼
                   ┌─────────────────────────────┐
                   │ 12th-Order Zero-Phase Filter│
                   │  & Linear RMS Calibrator    │
                   └──────────────┬──────────────┘
                                  ▼
                   ┌─────────────────────────────┐
                   │ Synthesized accelDFT (0.3s) │
                   │  16 kHz, Unit RMS = 1.0     │
                   └─────────────────────────────┘
```

---

## 🔬 Phase 2.4 Architecture & Technical Highlights

Phase 2.4 resolves the six core bottlenecks identified in earlier phases (envelope plateaus, low-frequency centroid pile-up, stopband leakage, modal peak blurring, phase decoherence, and clipping spikes):

1. **Band-Matched Vectorized 1-Sample FFT Alignment & In-Phase Polarity Locking (`preprocess_sliding.py`):**
   - Filters 1.0s audio to tactile frequencies (`30–800 Hz` and `30–350 Hz`), stripping high-frequency air-hiss before alignment.
   - Computes all 1-sample cross-correlation offsets in $\mathcal{O}(N \log N)$ via FFT across 64, 256, and 1024 RMS envelopes and waveform phase.
   - Locks accelerometer polarity in-phase directly in the preprocessed `.pt` tensors.

2. **Acoustic-Guided Hybrid Coherent + Dual-Resolution Decoder (`SlidingSEANetDecoder`):**
   - **Unbiased Spectral Tilt Conditioning:** Combines z-score peak contrast with unnormalized relative dB tilt and low-frequency energy ratios.
   - **40-Band Filterbank in `[28, 800 Hz]` with Global Per-Band Normalization:** Eliminates local-RMS carrier whitening to preserve rapid 4–16 ms friction micro-bursts.
   - **Floor Removal & Learned Spectral Tilt Gate:** Removes the $+0.05$ ($-26\text{ dB}$) floor to allow low-centroid surfaces to drop high frequencies by $>50\text{ dB}$, eliminating the 380–460 Hz pile-up.
   - **DifferentiableFineSpectralShaper (50 Welch + 232 rFFT bins):** Sharpens narrow $15.625\text{ Hz}$ resonant modal peaks.
   - **MultiScaleEnvelopeFusionTCN (16 channels):** Predicts band envelopes and post-filter master envelopes with multi-scale skip connections.
   - **Direct Phase-Coherent Waveform Residual Stream:** Translates tactile sub-bands for positive PCC and high SNR.
   - **Soft-Knee Crest Limiter + 12th-Order Zero-Phase FFT Bandpass:** Eliminates out-of-band stopband leakage above 800 Hz and eliminates clipping spikes before linear unit-RMS calibration.

3. **Exact Welch PSD + Outlier-Weighted Dual-Domain Loss (`SlidingTextureLoss`):**
   - Exact 1024-pt Welch PSD loss strictly matching `scipy.signal.welch`.
   - $10\times$ resonant modal peak-weighted PSD loss.
   - Outlier-weighted centroid loss ($2.2\times$ weight on $< 260\text{ Hz}$).
   - Stride-1 multi-scale envelope correlation loss.
   - Multi-band scale-invariant waveform PCC loss.

---

## 📊 Quantitative Benchmarks ($N = 200$ Stratified Validation Samples)

Evaluated strictly on independent, stratified validation instances across all 8 LMT material classes ($N = 200$ pairs):

### Overall Metric Progression Across Phases

| Metric | Phase 2.1 (Baseline) | Phase 2.2 (Stochastic) | Phase 2.3 (Acoustic-Guided) | **Phase 2.4 (Publication-Grade)** | Ideal |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Tactile Band (30–800Hz) LSD** | $4.94 \pm 0.85\text{ dB}$ | $3.46 \pm 1.06\text{ dB}$ | $3.17 \pm 0.83\text{ dB}$ (med $3.05$) | **$3.07 \pm 0.92\text{ dB}$** (med **$2.83\text{ dB}$**) | $\to 0.0$ |
| **PSD Cosine Similarity** | $0.546 \pm 0.135$ | $0.653 \pm 0.127$ | $0.735 \pm 0.127$ (med $0.766$) | **$0.757 \pm 0.129$** (med **$0.785$**) | $\to 1.0$ |
| **Peak Amplitude Ratio** | $0.468 \pm 0.147$ | $1.005 \pm 0.192$ | $1.100 \pm 0.198$ (med $1.070$) | **$1.036 \pm 0.169$** (med **$1.011$**) | $\to 1.0$ |
| **RMS Energy Ratio** | $0.697 \pm 0.069$ | $0.997 \pm 0.098$ | $0.973 \pm 0.016$ (med $0.973$) | **$0.990 \pm 0.015$** (med **$0.987$**) | $\to 1.0$ |
| **Tactile Band Energy Ratio** | $0.663 \pm 1.029$ | $0.841 \pm 0.201$ | $0.909 \pm 0.086$ (med $0.904$) | **$1.048 \pm 0.085$** (med **$1.039$**) | $\to 1.0$ |
| **Envelope Corr (64 / 4ms)** | $0.387 \pm 0.261$ | $0.277 \pm 0.138$ | $0.365 \pm 0.133$ (med $0.350$) | **$0.402 \pm 0.151$** (med **$0.385$**) | $\to 1.0$ |
| **Envelope Corr (256 / 16ms)** | $0.530 \pm 0.323$ | $0.473 \pm 0.190$ | $0.533 \pm 0.192$ (med $0.543$) | **$0.504 \pm 0.216$** (med **$0.510$**) | $\to 1.0$ |
| **Envelope Corr (1024 / 64ms)** | $0.620 \pm 0.391$ | $0.617 \pm 0.288$ | $0.634 \pm 0.295$ (med $0.710$) | **$0.635 \pm 0.288$** (med **$0.713$**) | $\to 1.0$ |
| **Pearson Correlation (PCC)** | $+0.0004 \pm 0.0646$ | $-0.0038 \pm 0.0394$ | $+0.0386 \pm 0.0707$ | **$+0.0668 \pm 0.0879$** (med **$+0.053$**, max **$+0.503$**) | $\to 1.0$ |
| **Spectral Centroid Delta** | $115.78 \pm 89.95\text{ Hz}$ | $96.06 \pm 62.60\text{ Hz}$ | $65.96 \pm 55.15\text{ Hz}$ | **$71.67 \pm 58.03\text{ Hz}$** (med **$60.74\text{ Hz}$**) | $\to 0.0$ |
| **Spectral Convergence** | $0.839 \pm 0.070$ | $0.782 \pm 0.152$ | $0.680 \pm 0.144$ | **$0.683 \pm 0.186$** (med **$0.672$**) | $\to 0.0$ |
| **Full-Band LSD (0–8 kHz)** | $12.48 \pm 13.25\text{ dB}$ | $5.58 \pm 1.65\text{ dB}$ | $3.59 \pm 1.77\text{ dB}$ | **$5.30 \pm 1.94\text{ dB}$** (12th-ord stopband) | $\to 0.0$ |
| **SPQI-Lite (Weighted LSD)** | $6.11 \pm 1.15$ | $5.87 \pm 1.04$ | $3.54 \pm 0.75$ | **$5.68 \pm 0.90$** | $\to 0.0$ |

### Per-Class Performance Breakdown ($N = 200$)

| Class | Count | PCC | RMSE | SNR (dB) | Peak Ratio | Tactile LSD (dB) | PSD CosSim | SPQI-Lite |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **C1 (Wood)** | 88 | +0.053 | 1.365 | -2.72 | 1.012 | 3.13 | 0.746 | 5.69 |
| **C2 (Paper/Fabric)** | 24 | **+0.110** | **1.314** | **-2.38** | 1.118 | 3.27 | 0.726 | 5.65 |
| **C3 (Plastic/Rubber)** | 28 | +0.086 | 1.343 | -2.55 | 1.093 | 3.20 | 0.750 | 5.48 |
| **C4 (Foam)** | 7 | +0.062 | 1.364 | -2.70 | 0.987 | 3.04 | 0.772 | 6.09 |
| **C5 (Carpet/Leather)** | 4 | +0.075 | 1.352 | -2.60 | 1.058 | **2.44** | **0.846** | **4.85** |
| **C6 (Stone/Ceramic)** | 31 | +0.056 | 1.366 | -2.71 | 0.976 | **2.77** | 0.782 | 5.77 |
| **C7 (Metal)** | 12 | +0.081 | 1.341 | -2.57 | 1.076 | 3.18 | 0.782 | 5.76 |
| **C8 (Glass)** | 6 | +0.040 | 1.374 | -2.78 | 1.054 | **2.50** | **0.825** | 6.01 |

---

## 📁 Repository Structure (`controlled-sliding` Branch)

```
haptics/
├── checkpoints/
│   └── best_sliding_decoder.pt          # Best Phase 2.4 model checkpoint (Epoch 146)
├── context/
│   ├── phase_2_2_sliding_overhaul_plan_and_context.md
│   ├── phase_2_3.md                     # Phase 2.3 overhaul specifications
│   ├── LMT Haptic Texture Database.pdf  # TUM CBM original dataset paper
│   └── Sound2Hap.pdf                    # Base architecture reference
├── evaluation_results_sliding/
│   ├── evaluation_aggregate_psd.png     # Aggregate PSD comparison across N=200
│   ├── evaluation_envelopes.png         # Multi-scale RMS envelope tracking
│   ├── evaluation_spectral.png          # Individual sample spectral profiles & centroids
│   └── evaluation_report.json           # Full JSON report containing all metrics
├── scripts/
│   ├── run_preprocessing_sliding.py     # Band-matched 1-sample FFT alignment
│   ├── train_sliding.py                 # Phase 2.4 training loop with AdamW + CosineAnnealing
│   ├── evaluate_sliding.py              # Full 200-sample validation benchmark suite
│   ├── inference_sliding.py             # Single-sample generation and 4-panel visualizer
│   ├── run_preprocessing.py             # Direct wrapper
│   ├── train.py                         # Direct wrapper
│   ├── evaluate.py                      # Direct wrapper
│   └── inference.py                     # Direct wrapper
├── src/
│   ├── data/
│   │   ├── preprocess_sliding.py        # SlidingPreprocessor (vectorized FFT alignment, unit-RMS)
│   │   └── dataset_sliding.py           # Stratified train/val split & PyTorch DataLoaders
│   ├── models/
│   │   ├── decoder_sliding.py           # SlidingSEANetDecoder (hybrid coherent + dual-resolution)
│   │   ├── loss_sliding.py              # Exact 1024-pt Welch PSD + outlier-weighted loss
│   │   └── encodec_wrapper.py           # Frozen 24kHz EnCodec audio encoder wrapper
│   └── evaluation/
│       └── metrics.py                   # Psychoacoustic LSD, SPQI-Lite, CosSim, Envelope Corr
├── inference_sliding_result.png         # 4-panel single-sample verification waveform
├── context.md                           # Continuous technical progress & benchmark log
└── README.md
```

---

## 🚀 Quickstart & Pipeline Workflow

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/ParthKhiriya/haptics.git
cd haptics

# Ensure you are on the controlled-sliding branch
git checkout controlled-sliding

# Install dependencies using uv
uv sync
```

### 2. Execution Pipeline
Each stage can be executed directly using `uv run python`:

```bash
# Step 1: Preprocess raw LMT continuous sliding pairs (vectorized FFT alignment & unit-RMS)
uv run python scripts/run_preprocessing_sliding.py

# Step 2: Train Phase 2.4 Hybrid Coherent Decoder (150 epochs)
uv run python scripts/train_sliding.py

# Step 3: Run comprehensive quantitative evaluation across all 200 validation samples
uv run python scripts/evaluate_sliding.py

# Step 4: Run single-sample inference and plot waveform synthesis
uv run python scripts/inference_sliding.py
```

All wrapper scripts (`python scripts/train.py`, etc.) route directly to the Phase 2.4 implementation.

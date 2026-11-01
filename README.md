# Cross-Modal Audio-to-Tactile Synthesis for Surface Textures (BTP)
## Continuous Controlled Sliding Interaction (`controlled-sliding` Branch)

> **B.Tech Project:** Understanding Multisensory Information to Build Perception of Object Properties  
> **Author:** Parth Khiriya (b23ee1051@iitj.ac.in)  
> **Institution:** Indian Institute of Technology Jodhpur (IITJ)  
> **Status:** Phase 2.3 Acoustic-Guided Coherent Sliding Synthesis (Benchmark Validated)

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
       │  Latent Sequences     │   │ & Sub-Band Envelopes      │
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
                   │ (25–1600 Hz, Unit Basis)    │
                   └──────────────┬──────────────┘
                                  ▼
                   ┌─────────────────────────────┐
                   │ Fused Temporal Envelope Gate│
                   │ & RMS Calibration Head      │
                   └──────────────┬──────────────┘
                                  ▼
                   ┌─────────────────────────────┐
                   │ Synthesized accelDFT (0.3s) │
                   │  16 kHz, Unit RMS = 1.0     │
                   └─────────────────────────────┘
```

---

## 🔬 Phase 2.3 Architecture & Technical Highlights

Phase 2.3 addresses the stochastic noise floor and EnCodec representation bottleneck through three primary innovations:

1. **Physics-Informed Dual-Band Envelope Alignment & Strict Unit-RMS Normalization:**
   - Aligns the 0.3s haptic vibration against the 1.0s audio window using multi-scale envelope cross-correlation combining low-band (`30–800 Hz` tool-tip vibration) and high-band (`800–6000 Hz` micro-asperity friction) envelopes at 2 ms (32-sample) resolution.
   - Enforces polarity consistency and strict unit-RMS (`rms_gt == 1.000`) after soft outlier compression (`4.2 * tanh(x / 4.2)`).

2. **Acoustic-Guided Coherent Decoder (`SlidingSEANetDecoder`):**
   - **Direct Raw-Audio Spectral Conditioning:** Extracts 64-bin log-power STFT spectral features (`20–4000 Hz`) directly from raw audio alongside EnCodec latents, breaking the speech-prior centroid bias.
   - **40-Band Filterbank with Butterworth Roll-Off:** Extends synthesis from 800 Hz to 1600 Hz with a 4th-order Butterworth lowpass gain above 800 Hz, eliminating stopband cliff artifacts.
   - **Deterministic Coherent Excitation Basis:** Replaces per-step Gaussian noise sampling with a fixed, buffer-registered unitary excitation basis (`seed=2026`) mixed with whitened acoustic sub-band excitation, removing STFT gradient speckle.
   - **Acoustic Sub-Band Envelope Fusion:** Dynamically gates and shapes the temporal envelope using learned and raw sub-band envelopes.

3. **Speckle-Free Multi-Resolution Welch & Wasserstein CDF Loss (`SlidingTextureLoss`):**
   - Eliminates noisy 2D STFT frame differences in favor of **Multi-Resolution Welch PSD Loss** (`n_fft = 256, 512, 1024, 2048`).
   - Optimizes **1D Spectral Wasserstein (Earth Mover's) Distance** on the cumulative distribution function (CDF) of normalized PSD, paired with differentiable centroid and spread loss.
   - Multi-scale Pearson envelope correlation loss (`win = 64, 128, 256, 512, 1024`).

---

## 📊 Quantitative Benchmarks ($N = 200$ Stratified Validation Samples)

Evaluated strictly on independent, stratified validation instances across all 8 LMT material classes ($N = 200$ pairs):

### Overall Metric Progression Across Phases

| Metric | Phase 2.1 (Baseline) | Phase 2.2 (Stochastic) | **Phase 2.3 (Acoustic-Guided)** | Ideal |
|:---|:---:|:---:|:---:|:---:|
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
| **Pearson Correlation (PCC)** | $+0.0004 \pm 0.0646$ | $-0.0038 \pm 0.0394$ | **$+0.0386 \pm 0.0707$** (max **$+0.503$**) | $\to 1.0$ |

### Per-Class Performance Breakdown ($N = 200$)

| Class | Count | PCC | RMSE | SNR (dB) | Peak Ratio | Tactile LSD (dB) | PSD CosSim | SPQI-Lite |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **C1 (Wood)** | 88 | +0.042 | 1.362 | -2.71 | 1.086 | 3.23 | 0.718 | 3.59 |
| **C2 (Paper/Fabric)** | 24 | +0.036 | 1.355 | -2.67 | 1.153 | 3.40 | 0.707 | 3.64 |
| **C3 (Plastic/Rubber)** | 28 | +0.039 | 1.364 | -2.69 | 1.144 | 3.27 | 0.740 | 3.55 |
| **C4 (Foam)** | 7 | +0.018 | 1.380 | -2.79 | 1.036 | **2.99** | **0.802** | 3.63 |
| **C5 (Carpet/Leather)** | 4 | **+0.070** | **1.345** | **-2.57** | 1.144 | **2.48** | **0.829** | **2.58** |
| **C6 (Stone/Ceramic)** | 31 | +0.038 | 1.369 | -2.73 | 1.059 | **2.80** | **0.773** | **3.40** |
| **C7 (Metal)** | 12 | +0.022 | 1.374 | -2.79 | 1.098 | 3.52 | 0.726 | 3.71 |
| **C8 (Glass)** | 6 | +0.039 | 1.358 | -2.68 | 1.144 | **2.79** | **0.755** | **3.41** |

---

## 📁 Repository Structure (`controlled-sliding` Branch)

```
haptics/
├── checkpoints/
│   └── best_sliding_decoder.pt          # Best Phase 2.3 model checkpoint (Epoch 147)
├── context/
│   ├── phase_2_2_sliding_overhaul_plan_and_context.md
│   ├── phase_2_3.md                     # Complete Phase 2.3 overhaul specifications
│   ├── LMT Haptic Texture Database.pdf  # TUM CBM original dataset paper
│   └── Sound2Hap.pdf                    # Base architecture reference
├── evaluation_results_sliding/
│   ├── evaluation_aggregate_psd.png     # Aggregate PSD comparison across N=200
│   ├── evaluation_envelopes.png         # Multi-scale RMS envelope tracking
│   ├── evaluation_spectral.png          # Individual sample spectral profiles & centroids
│   └── evaluation_report.json           # Full JSON report containing all metrics
├── scripts/
│   ├── run_preprocessing_sliding.py     # Physics-informed dual-band envelope alignment
│   ├── train_sliding.py                 # Phase 2.3 training loop with AdamW + CosineAnnealing
│   ├── evaluate_sliding.py              # Full 200-sample validation benchmark suite
│   ├── inference_sliding.py             # Single-sample generation and 4-panel visualizer
│   ├── run_preprocessing.py             # Direct wrapper
│   ├── train.py                         # Direct wrapper
│   ├── evaluate.py                      # Direct wrapper
│   └── inference.py                     # Direct wrapper
├── src/
│   ├── data/
│   │   ├── preprocess_sliding.py        # SlidingPreprocessor (dual-band alignment, unit-RMS)
│   │   └── dataset_sliding.py           # Stratified train/val split & PyTorch DataLoaders
│   ├── models/
│   │   ├── decoder_sliding.py           # SlidingSEANetDecoder (40-band coherent filterbank)
│   │   ├── loss_sliding.py              # Multi-Resolution Welch PSD + Wasserstein CDF loss
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
# Step 1: Preprocess raw LMT continuous sliding pairs (dual-band alignment & unit-RMS)
uv run python scripts/run_preprocessing_sliding.py

# Step 2: Train Phase 2.3 Acoustic-Guided Coherent Decoder (150 epochs)
uv run python scripts/train_sliding.py

# Step 3: Run comprehensive quantitative evaluation across all 200 validation samples
uv run python scripts/evaluate_sliding.py

# Step 4: Run single-sample inference and plot waveform synthesis
uv run python scripts/inference_sliding.py
```

All wrapper scripts (`python scripts/train.py`, etc.) are also available and route directly to the Phase 2.3 implementation.

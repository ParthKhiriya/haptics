# Project Context — Audio-to-Haptic Cross-Modal Synthesis

> **Last Updated:** 2026-10-23 11:12 IST  
> **Project:** BTP — Understanding Multisensory Information to Build Perception of Object Properties  
> **Author:** ParthKhiriya (b23ee1051@iitj.ac.in)  
> **Current Task:** Controlled Tap Results Assessment & Transition to Phase 2

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
|-------|-------------|------|-------------|
| **Sound2Hap** (Li & Seifi, CHI '26) | CNN autoencoder (EnCodec encoder + SEANet decoder) | Audio → vibrotactile | Frozen EnCodec encoder, SEANet decoder with LSTM + residual upsampling, LeakyReLU, no tanh output. Trained with human perceptual ratings. |
| **HapticGen** (Sung et al., CHI '25) | Autoregressive transformer + EnCodec | Text → vibrotactile | Leverages EnCodec neural audio codec |
| **VibNet** | Deep learning framework | Predict roughness/emotion from haptic | Biomimetic approach |
| **Cluster Haptic Texture Dataset** (Eguchi, 2026) | — | Dataset: 118 surfaces | 3-axis machine, controlled velocity/direction |
| **Devillard's Dataset** (Imperial, 2024-25) | — | Dataset: 10 surfaces | Bare-finger, multimodal |
| **Haptipedia** (Agrawal et al.) | — | Device corpus | Taxonomy for haptic device design |

---

## 3. Dataset — LMT Haptic Texture Database (TUM CBM)

### 3.1 Taxonomy and File Naming Structure
Every `.mat` file in `data/LMT_FinalDatabase/CBM_FinalDatabase/` follows the standard format:
`C<class>_S<subclass>_M<material>_P<processing>_X<composition>_I<instance>.mat`

- **C (Class 1–8):**
  - C1: Wood (Softwood, Hardwood, Barks, Celluloids, Crafted Wood)
  - C2: Paper / Fabric / Biodegradable (Natural fibers, Fur, Leather)
  - C3: Polymers / Plastics (Thermoplasts, Thermosets, Elastomers, Carbon fibers)
  - C4: Ceramics (Earthenware, Stoneware, High-tech ceramics)
  - C5: Glasses (Crafted glasses, Natural glasses)
  - C6: Stones (Igneous, Sedimentary, Metamorphic, Cement, Minerals)
  - C7: Metals (Ferrous, Non-ferrous, Alloys, Metal fibers)
  - C8: Composites (Polymer-Stone, Polymer-Metal, Wood-Stone)
- **S (Subclass):** Geological, biological, or chemical category.
- **M (Material ID):** Specific material specimen (e.g., spruce, pine, granite, steel).
- **P (Processing / Shape):**
  - Wood: P1 veneered tangential, P2 cross-section, P3 sliced, P4 natural tangential, P5 natural cross-section.
  - Stone: P1 stacked, P2 polished, P3 pointed, P4 sanded, P5 embossed.
  - Metals: P1 steamrolled, P2 polished, P3 casted, P4 perforated.
- **X (Composition / Moisture / Finish):**
  - Wood: moisture percentage.
  - Polymers/Metals: purity percentage.
  - Glass: opacity level.
  - Composites/Abrasives: grit size (X4=P40, X8=P80, X24=P240, X60=P600).
- **I (Instance 1–5):** 5 repeated independent measurement passes per surface.

Total: 1,115 valid material files across all 8 classes.

### 3.2 Modality & Interaction Decomposition
Each `.mat` file bundles multiple distinct physical interactions and sensing modalities:

1. **Controlled Tap Interaction (Impact Contact)**:
   - `soundTap`: 5.0 seconds @ 44,100 Hz (220,500 samples) contact acoustic tap.
   - `accelTap`: 1.5 seconds @ 10,000 Hz (15,003 samples) impact acceleration.
   - `normalForce`: 1.5 seconds @ 10,000 Hz (15,003 samples) dynamic normal contact force.
   - `frictionForce`: 1.5 seconds @ 10,000 Hz (15,003 samples) dynamic lateral force.
   - `soundTapSteelTooltip` & `accelTapSteelTooltip`: tool-tip variant using customized stainless-steel tip.
2. **Continuous Sliding / Stroking Interaction (Surface Texture Friction)**:
   - `sound` / `soundSteelTooltip`: 1.0 second @ 44,100 Hz (44,100 samples) sliding friction sound.
   - `accelDFT` / `accelDFTSteelTooltip`: 0.3 seconds @ 10,000 Hz (3,000 samples) sliding vibration, reduced from 3D to 1D via Kuchenbecker's `DFT321` algorithm.
3. **Mechanical Compliance / Flexibility**:
   - `pressure`, `pressureFolding`, `foldingAngle`: 12,500 samples measuring bending resistance.
4. **Thermal, Electrical & Optical Sensors**:
   - `temperatureData`: 2,240 samples measuring heat dissipation into the material.
   - `metalData`: 3,000 samples measuring eddy-current inductive response.
   - `reflectance`: 3,000 samples measuring optical reflectance.
5. **Visual Photography**:
   - `displayImage`, `macroImage`, `illuminatedImage`: high-resolution $(4032 \times 3024 \times 3)$ RGB photography.

### 3.3 Disambiguation: Controlled vs. Uncontrolled vs. Tap vs. Sliding
There are two orthogonal distinctions in the LMT database:

| Dimension | Concept 1 | Concept 2 |
|---|---|---|
| **Actuator / Operator** | **Controlled (Robotic / Machine):** Performed by the **Texplorer / Phantom Omni** robotic arm. Velocity, force, and stylus orientation are electronically controlled by PID controllers and a motorized rotary table. | **Uncontrolled / Unconstrained (Human Hand):** Performed by a human operator holding the stylus naturally in freehand movement (without fixed speed, fixed force, or fixed angle). |
| **Physical Motion** | **Tap (Impact):** Perpendicular vertical impact to study contact hardness, elasticity, and impact transients (`soundTap`, `accelTap`, `normalForce`). | **Continuous Sliding (Friction):** Lateral dragging across the surface to study texture ridges, roughness, and friction (`sound`, `accelDFT`). |

**Key Clarification:**
- Controlled $\ne$ Tap. Controlled data includes **BOTH** robotic tapping AND robotic sliding.
- Uncontrolled $\ne$ Sliding. Uncontrolled data refers to **freehand human movement** (e.g. 5 linear and 5 circular hand exploratory strokes).
- Our current model operates on **Texplorer Robotic Tap Data** (`soundTap` $\to$ `accelTap`). It contains **NO human hand tapping**.

---

## 4. Architecture — Sound2Hap Implementation (v2.1)

### Pipeline Overview
```
soundTap (44100 Hz, 5.0s)
  → Peak detection & 1.0s window extraction
  → Resample to 16 kHz (16000 samples)
  → Peak-normalized to [-1, 1]
  → EnCodec Encoder (frozen, resamples 16k -> 24k)
  → Latent embeddings [B, 128, 75]
  → SEANet Decoder (LSTM + ResNet upsampling + hardtanh(-6, 6))
  → Predicted Haptic [B, 1, 16000]
  → Supervised by accelTap (bandpassed 20-500Hz, peak-aligned, 16kHz, calibrated [-5, 5])
```

### Model & Loss Design
- **EnCodec Wrapper:** Resamples from `16000` to `24000` Hz. Latents extracted with gradients detached.
- **SEANet Decoder:** LSTM with 0.1 dropout, 4 upsample blocks, final projection through `F.hardtanh(x, -6.0, 6.0)` to eliminate tanh saturation and provide 1σ headroom over target bounds.
- **Sound2HapLossV2:**
  - Multi-resolution tactile-band focused STFT loss (50–300 Hz focus)
  - Multi-scale RMS envelope loss with Pearson correlation
  - Log-Mel spectrogram loss (f_min=20, f_max=2000, 16kHz)
  - Time-domain MSE & L1 loss

---

## 5. Full Dataset Quantitative Evaluation (N = 186 Validation Samples)

### 5.1 Dataset-Wide Summary Metrics
- **Envelope Correlation:**
  - `Envelope Corr (64)`: **0.9275 ± 0.1235** (median **0.961**)
  - `Envelope Corr (256)`: **0.9577 ± 0.1180** (median **0.985**)
  - `Envelope Corr (1024)`: **0.9705 ± 0.1204** (median **0.996**)
- **Amplitude & Energy Conservation:**
  - `Peak Amplitude Ratio`: **1.1688 ± 0.1876** (median **1.200**)
  - `RMS Energy Ratio`: **0.8264 ± 0.1524** (median **0.865**)
  - `Tactile Band Energy Ratio (50–300 Hz)`: **0.8496 ± 0.2285** (median **0.848**)
- **Spectral Fidelity:**
  - `Tactile Band (50–300 Hz) LSD`: **3.4478 ± 4.6254 dB** (median **2.48 dB**)
  - `PSD Cosine Similarity`: **0.8100 ± 0.1696**
  - `Spectral Centroid Delta`: **36.95 ± 34.68 Hz**
- **Time Domain:**
  - `Pearson Correlation (PCC)`: **0.2327 ± 0.3901**
  - `RMSE`: **0.5680 ± 0.1548**
  - `SNR (dB)`: **-0.5199 ± 2.1837 dB**

### 5.2 Per-Class Performance Breakdown

| Class | Count | PCC | RMSE | SNR (dB) | Peak Ratio | Tactile LSD (dB) | CosSim | Interpretation |
|---|---|---|---|---|---|---|---|---|
| **C1 (Wood)** | 85 | 0.148 | 0.598 | -1.04 | 1.159 | 3.95 | 0.771 | Resonant high-frequency chattering & micro-phase cancellation |
| **C2 (Paper/Fabric)** | 26 | 0.340 | 0.578 | -0.08 | 1.200 | **1.86** | **0.942** | High spectral fidelity, smooth damping |
| **C3 (Plastic/Rubber)**| 18 | 0.479 | 0.492 | +0.63 | 1.200 | **1.72** | **0.932** | Strong shape and frequency tracking |
| **C4 (Foam)** | 3 | 0.339 | 0.570 | -0.43 | 1.200 | **1.82** | **0.908** | Strong tactile match, damped transient |
| **C5 (Carpet/Leather)**| 4 | **0.655** | **0.403** | **+2.04** | 1.200 | 2.68 | **0.946** | Highest correlation & SNR |
| **C6 (Stone/Ceramic)** | 39 | 0.070 | 0.591 | -1.17 | 1.141 | 4.81 | 0.697 | Rigid bounce dynamics, phase jitter |
| **C7 (Metal)** | 7 | **0.634** | **0.395** | **+2.18** | 1.200 | 2.21 | **0.918** | Excellent ringing profile & energy match |
| **C8 (Glass)** | 4 | **0.600** | **0.439** | **+1.54** | 1.200 | **1.71** | **0.925** | High fidelity across both domains |

---

## 6. Execution & Implementation Status

- [x] Preprocessing script overhauled (`src/data/preprocess.py`)
- [x] Dataset updated: removed double z-score re-normalization (`src/data/dataset.py`)
- [x] Decoder updated: replaced `tanh * scale` with `hardtanh(-6, 6)` (`src/models/decoder.py`)
- [x] Evaluation metrics module created (`src/evaluation/metrics.py`)
- [x] Evaluation runner script created (`scripts/evaluate.py`)
- [x] Inference script upgraded with objective metric calculation (`scripts/inference.py`)
- [x] Integration checks & unit tests passed
- [x] Model retrained to 100 epochs on corrected pipeline (`scripts/train.py`)
- [x] Full validation set evaluated across all 8 classes (186 samples)
- [x] Full dataset architecture & physical parameters analyzed
- [x] Conceptual disambiguation: Controlled vs Uncontrolled vs Tap vs Sliding clarified
- [x] Controlled Tap Synthesis Phase: Validated & Finalized as solid baseline
- [ ] Phase 2: Continuous Sliding Texture Synthesis (`sound` -> `accelDFT`) or Multimodal Expansion (TexTUM Vision + Audio)

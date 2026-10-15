# Project Context — Audio-to-Haptic Cross-Modal Synthesis

> **Last Updated:** 2026-10-09 12:54 IST  
> **Project:** BTP — Understanding Multisensory Information to Build Perception of Object Properties  
> **Author:** ParthKhiriya (b23ee1051@iitj.ac.in)  
> **Current Task:** Audio → Haptic/Tactile signal generation using Sound2Hap architecture

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

## 3. Dataset — LMT Haptic Texture Database (TUM)

### Selection Rationale
Selected over Cluster (118 surfaces) and Devillard (10 surfaces) datasets for:
- **184 materials** across 8 classes
- Both controlled (Texplorer tool) and uncontrolled (human hand) data
- Multiple instances per material
- Synchronized audio + acceleration + force + images

### Data Location & Structure
```
data/LMT_FinalDatabase/CBM_FinalDatabase/
├── C1/ (443 .mat files) — Wood
├── C2/ (120 .mat files) — Paper/Fabric
├── C3/ (176 .mat files) — Plastic/Rubber
├── C4/ (40 .mat files)  — Foam
├── C5/ (55 .mat files)  — Carpet/Leather
├── C6/ (176 .mat files) — Stone/Ceramic
├── C7/ (70 .mat files)  — Metal
└── C8/ (35 .mat files)  — Glass

Total: 1,115 .mat files (avg ~75 MB each)
```

### Other Data Folders
- `data/TexTUM_Adjusted_193200/` — 193,134 .jpg texture images (visual, future use)
- `data/TexTUM___64400_DB_Paper/` — 64,400 .jpg texture images (visual, future use)
- `data/processed_old/` — 929 .npy pairs from previous buggy preprocessing (backed up)
- `data/processed/` — Currently generating from controlled Tap data only

### Controlled vs Uncontrolled Data

**Current Focus:** Controlled (Tap) data only.
- Audio: `soundTap` (44100 Hz native)
- Haptic: `accelTap` (10000 Hz native)
- Processed window: 1.0 second aligned around peak event at **16,000 Hz**

---

## 4. Architecture — Sound2Hap Implementation (v2)

### Pipeline Overview
```
soundTap (44100 Hz, 5.0s)
  → Peak detection & 1.0s window extraction
  → Resample to 16 kHz (16000 samples)
  → Peak-normalized to [-1, 1]
  → EnCodec Encoder (frozen, resamples 16k -> 24k)
  → Latent embeddings [B, 128, 75]
  → SEANet Decoder (LSTM + ResNet upsampling + tanh * scale)
  → Predicted Haptic [B, 1, 16000]
  → Supervised by accelTap (bandpassed 20-500Hz, peak-aligned, 16kHz, z-scored)
```

### Model & Loss Design
- **EnCodec Wrapper:** Resamples from `16000` to `24000` Hz. Latents extracted with gradients detached.
- **SEANet Decoder:** LSTM with 0.1 dropout, 4 upsample blocks, output constrained via `tanh(x) * output_scale`. Interpolated to 16,000 samples.
- **Sound2HapLossV2:**
  - Multi-resolution tactile-band focused STFT loss (50–300 Hz focus)
  - Multi-scale RMS envelope loss with Pearson correlation
  - Log-Mel spectrogram loss (f_min=20, f_max=2000, 16kHz)
  - Time-domain MSE & L1 loss

---

## 5. Execution & Implementation Status

- [x] Preprocessing script overhauled (`src/data/preprocess.py`)
- [x] Dataset & DataLoader updated (`src/data/dataset.py`)
- [x] Encodec wrapper updated with 16k->24k fix (`src/models/encodec_wrapper.py`)
- [x] Decoder bounded and aligned (`src/models/decoder.py`)
- [x] Loss redesigned with tactile emphasis (`src/models/loss.py`)
- [x] Training & inference scripts adapted (`scripts/train.py`, `scripts/inference.py`)
- [x] Unit tests passed on CUDA (`encoder`, `decoder`, `loss`)
- [x] Preprocessing on controlled data launched (`scripts/run_preprocessing_v2.py`)
- [ ] Model retraining on cleaned dataset

# Cross-Modal Audio-to-Tactile Synthesis for Surface Textures (BTP)

> **B.Tech Project:** Understanding Multisensory Information to Build Perception of Object Properties  
> **Author:** Parth Khiriya (b23ee1051@iitj.ac.in)  
> **Institution:** Indian Institute of Technology Jodhpur (IITJ)  

---

## 📖 Overview

This project investigates cross-modal perception and generative modeling between **audio** and **tactile (vibrotactile)** modalities during physical interactions with textured surfaces.

Using surface contact audio recorded with tool-tip interaction, the system leverages a frozen neural audio codec representation (**EnCodec**, Meta AI) and lightweight convolutional decoders (**SEANet**-derived architectures) to synthesize high-fidelity, perceptually aligned 3-axis vibrotactile acceleration signals.

---

## 🌿 Branching Architecture & Modality Isolation

To ensure maximum cleanliness, reproducibility, and modularity, **each interaction type is maintained in a dedicated, self-contained branch**. Any branch can be cloned and run as an independent project:

| Branch | Interaction Type | Input Audio | Output Tactile | Status |
| :--- | :--- | :--- | :--- | :--- |
| **`main`** | **Latest Advancements** | Unified / Latest | Unified / Latest | 🚀 Active |
| **`controlled-tap`** | Controlled Tap Interaction | `soundTap` (16 kHz) | `accelTapDFT` (16 kHz, 1.0s) | ✅ Finalized Baseline |
| **`controlled-sliding`** | Continuous Sliding Textures | `sound` (16 kHz) | `accelDFT` (16 kHz, 0.3s) | 🔄 In Progress |
| **`uncontrolled-sliding`** | Freehand Sliding Interaction | Continuous audio | Freehand acceleration | 📋 Planned |
| **`video-to-tactile`** | Vision + Audio Multimodal | Video frames + sound | Surface acceleration | 📋 Planned |

---

## 📁 Repository Structure

```
haptics/
├── checkpoints/             # Model weights (best_sean_decoder.pt, best_sliding_decoder.pt)
├── context/                 # Reference papers, database guides, and BTP documentation
├── dft321/                  # MATLAB reference implementation for 3-axis DFT acceleration
├── evaluation_results/      # Spectral, envelope, and perceptual metrics & figures
├── scripts/
│   ├── run_preprocessing.py # Dataset extraction and Butterworth tactile filtering
│   ├── train.py             # Training loop with multi-scale STFT & envelope loss
│   ├── evaluate.py          # Full quantitative evaluation suite (LSD, envelope, SPQI)
│   └── inference.py         # Visual and audio-tactile generation test
├── src/
│   ├── data/                # Data preprocessors and PyTorch DataLoaders
│   ├── models/              # EnCodec encoder wrapper, SEANet decoders, composite losses
│   ├── evaluation/          # Psychoacoustic, time-domain, and frequency-domain metrics
│   └── utils/               # Python translation of DFT321 coordinate conversions
├── pyproject.toml           # Project dependencies managed via uv
├── context.md               # Continuous project progress log and technical details
└── README.md
```

> **Note on Agent & Diagnostic Scripts:**  
> One-off diagnostic, inspection, and verification scripts (`extract_and_inspect.py`, `scripts/check_keys.py`, `scripts/test_*.py`) are ignored via `.gitignore` and kept strictly local to maintain a clean GitHub repository.

---

## 🚀 Quickstart

### 1. Environment Setup
The project uses `uv` for fast, reproducible Python packaging:

```bash
# Clone the repository
git clone https://github.com/ParthKhiriya/haptics.git
cd haptics

# Install dependencies using uv
uv sync
```

### 2. Switching to an Interaction Branch
To inspect or run a specific interaction as an isolated project:

```bash
# For Controlled Tap:
git checkout controlled-tap

# For Continuous Controlled Sliding:
git checkout controlled-sliding
```

### 3. Pipeline Workflow
Each branch follows the same canonical 4-step pipeline:

```bash
# Step 1: Preprocess raw LMT database files
python scripts/run_preprocessing.py

# Step 2: Train the interaction decoder
python scripts/train.py

# Step 3: Run comprehensive evaluation
python scripts/evaluate.py

# Step 4: Run single-sample inference and plot synthesis
python scripts/inference.py
```

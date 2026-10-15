# Project Context and Status Report

**Target System:** Google Antigravity
**Project Type:** Bachelor of Technology Project (BTP)
**Project Title:** Understanding Multisensory Information to Build Perception of Object Properties

## 1. Basic Context & Project Objectives

This BTP aims to investigate the relationships between different sensing modalities (vision, sound, and haptics) when interacting with rigid and non-rigid objects, focusing specifically on properties like hardness and roughness.

The core research objectives are:

1. **Cross-Modal Learning:** Designing a generative AI-based model to predict touch (haptic) signals from visual and sound modalities.

2. **Multimodal Interaction:** Investigating the relative weights of different modalities during interactions with textured surfaces.

3. **Physical Features:** Studying perceptually validated physical features corresponding to object attributes like hardness and roughness.

4. **Perceptual Dimensions:** Investigating the perceptual space of textured surfaces during tool-tip interactions.

## 2. Literature Review: Research Papers Studied

To build the foundation for this project, we have analyzed several state-of-the-art research papers in the haptics and multimodal AI space. The key papers include:

* **Sound2Hap:** *Learning Audio-to-Vibrotactile Haptic Generation from Human Ratings* (Li and Seifi, CHI '26) - Explores generating vibrations from environmental sounds using a CNN-based autoencoder trained on human ratings.

* **HapticGen:** *Generative Text-to-Vibration Model for Streamlining Haptic Design* (Sung et al., CHI '25) - Details an autoregressive transformer model leveraging EnCodec to convert text prompts into vibrotactile signals.

* **Cluster Haptic Texture Dataset:** *Haptic texture Dataset with Varied Velocity-Direction Sliding Contacts* (Eguchi et al., 2026) - Discusses a multimodal dataset recorded using a 3-axis machine for controlled sliding velocity and direction.

* **Tactile, Audio, and Visual Dataset:** *During Bare Finger Interaction with Textured Surfaces* (Devillard et al., 2024/2025) - Presents concurrent haptic, audio, and visual signals recorded during bare-finger interactions with varied surfaces.

* **VibNet:** *A Biomimetic Framework for Predicting Roughness and Emotion Conveyed through Vibrations* - Explores deep learning for predicting user perception of haptic signals.

* **Haptipedia** (Agrawal et al.) - Explores device corpus and taxonomy for haptic device design.

## 3. Datasets Explored & Selected

During the initial phase, we evaluated several datasets to train our cross-modal models:

* **Cluster Haptic Texture Dataset:** 118 textured surfaces with controlled sliding conditions (audio, acceleration, force, visual).

* **Devillard's Multi-modal Dataset (Imperial College London):** 10 textured surfaces explored via bare fingers (stereoscopic images, position, audio, friction vibrations).

* **LMT Haptic Texture Dataset(Technical University of Munich)(184 Materials): This dataset was selected by us, it contains 184 materials data, with all the required readings, it has controlled as well as uncontrolled data, data at different velocities and force.** 

**Selected Dataset:** For our current primary task, we have selected the **LMT Haptic Texture Dataset(TUM).** 

## 4. Current Task & Implementation State

**First Major Task: Audio to Haptic Conversion**
We are currently tackling the first major objective of our cross-modal learning goal: generating accurate, perceptually meaningful haptic (touch) signals directly from audio input.

**Architecture in Use:**
We have adopted the **Sound2Hap** architecture for this phase. It is a CNN-based autoencoder designed to learn mappings from audio waveforms to vibration signals.

**Current Progress:**

* We have written the initial code implementation for the model.

* Data pipelines for audio and vibration signals have been set up.

* Initial training and inference passes have been executed.

## 5. Current Roadblocks & Problems Faced

While the codebase is functioning, the generated haptic outputs are heavily degraded. We are currently trying to debug and resolve the following specific issues:

1. **Very High Noise in the Output:** The generated haptic signals contain a significant amount of artifacting and high-frequency noise that does not map to a realistic tactile sensation. It currently feels muddled rather than distinct.

2. **Amplitude Mismatch:** The intensity (amplitude) of the predicted haptic signals does not correctly align with the input audio or the ground-truth target vibrations. The peaks and troughs are disproportionate, leading to an inaccurate representation of impact or texture.

## 6. Request for Antigravity

We need assistance in analyzing our current Sound2Hap implementation to resolve the high noise and amplitude mismatch issues. Specifically, we are looking for:

* Architectural or loss-function tweaks to force the model to respect amplitude boundaries.

* Signal processing recommendations (e.g., pre-processing or post-generation filtering) to clean up the noise and align the output with human tactile sensitivity (e.g., concentrating on the 50Hz-300Hz optimal range).
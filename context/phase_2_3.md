\# ANTIGRAVITY IMPLEMENTATION PLAN: Phase 2.3 — Acoustic-Guided Coherent Sliding Synthesis

\#\# 1\. Executive Summary & Root Causes of the Phase 2.2 Plateau  
In Phase 2.2, we resolved the \`0.46x\` amplitude squash (\`Peak Ratio \= 1.005\`, \`RMS Ratio \= 0.997\`), eliminated \`ConvTranspose1d\` comb notches, and reduced \`Tactile Band LSD\` from \`4.94 dB\` to \`3.46 dB\`. However, training plateaued at Epoch 34 (\`Val Loss: 10.0467\`) and triggered early stopping at Epoch 59 due to five specific bottlenecks:

1\. \*\*Stochastic \`torch.randn\` STFT Speckle Floor & Envelope Jitter:\*\* Sampling fresh Gaussian noise \`torch.randn\` on every forward pass creates random frame-to-frame STFT bin fluctuations (speckle variance) that create an irreducible loss floor in \`\_multi\_res\_stft\_loss\`, stalling gradient descent around Epoch 34\. It also injects random short-term bumps into the predicted envelope (\`Env Corr (64) \= 0.277\`, \`Env Corr (256) \= 0.473\`).  
2\. \*\*EnCodec Information Bottleneck (\`Centroid Delta \= 96.06 Hz\`):\*\* The decoder only received frozen EnCodec latents \`latents\` and had no direct access to the aligned raw audio \`audio\[:, :, :4800\]\`. Because EnCodec was trained on speech/music, it collapses diverse surface friction spectra into a narrow \`350–500 Hz\` predicted centroid range.  
3\. \*\*Stopband Cliff Above \`800 Hz\` (\`Full-Band LSD \= 5.58 dB\`):\*\* Placing all 32 filterbank bands strictly below \`800 Hz\` caused a vertical \`-115 dB/Hz\` cliff between \`800–2000 Hz\`, whereas the 4th-order Butterworth ground truth rolls off smoothly at \`-80 dB/decade\`.  
4\. \*\*Uncorrelated Unit-Variance Noise Penalty (\`SNR \= \-3.01 dB\`, \`RMSE \= 1.373\`):\*\* Two uncorrelated unit-RMS signals ($\rho =0,\sigma =1$) mathematically yield $RMSE=\sqrt{2}\approx 1.414$ and $SNR=-3.01dB$. Using acoustic-seeded coherent excitation and optimal MMSE energy scaling improves both waveform coherence and SNR.  
5\. \*\*Sub-Band Misalignment & \`rms\_gt \< 1.0\` Outliers in \`preprocess\_sliding.py\`:\*\* Aligning \`0.3s\` audio using full-band \`20–7800 Hz\` energy (dominated by \`\>2 kHz\` air hiss) instead of the \`30–800 Hz\` mechanical sub-band caused envelope misalignment, and \`min(1.0/rms, 5.0/peak)\` left spiky ground-truth samples with \`rms\_gt \= 0.57–0.85\`.

\---

\#\# 2\. Step-by-Step Implementation Plan & Complete Code Specifications

\#\#\# STEP 1: Upgrade \`src/data/preprocess\_sliding.py\` (Sub-Band Envelope Alignment & Strict Unit-RMS Target)

\#\#\#\# What to Change:  
1\. \*\*Physics-Informed Dual-Band Envelope Alignment (\`\_align\_audio\_window\`):\*\*  
   \- Extract two complementary envelopes from the \`1.0s\` audio (\`16 kHz\`):  
     \- \`env\_low\`: RMS envelope of the \`30–800 Hz\` bandpass-filtered audio (captures direct mechanical tool-tip vibration).  
     \- \`env\_high\`: RMS envelope of the \`800–6000 Hz\` bandpass-filtered audio (captures micro-asperity friction bursts).  
   \- Combine \`0.6 \* zscore(env\_low) \+ 0.4 \* zscore(env\_high)\` and slide a \`4800\`-sample (\`0.3s\`) window with a fine \`32\`-sample (\`2 ms\`) step across \`\[0.08s, 0.68s\]\` to find the exact segment that maximizes correlation with the \`0.3s\` \`accelDFT\` envelope (at both \`256\` and \`64\` window scales).  
   \- Also check the sign of the waveform correlation between the \`30–800 Hz\` audio sub-band and \`accelDFT\`, and flip polarity if negative so the macro-transients are in phase.  
2\. \*\*Strict Unit-RMS Ground-Truth Normalization:\*\*  
   \- Soft-clip extreme \`accelDFT\` outliers using \`3.8 \* np.tanh(haptic\_norm / 3.8)\` and then \*\*re-normalize by RMS\*\* (\`haptic\_norm / rms\_post\`) followed by \`np.clip(haptic\_norm, \-5.0, 5.0)\` so \*\*every single ground-truth file has \`rms\_gt \== 1.000\`\*\* (eliminating \`rms\_gt \= 0.57\` outliers).

\#\#\#\# Drop-In Replacement for \`src/data/preprocess\_sliding.py\`:  
\`\`\`python  
import re  
from pathlib import Path  
import numpy as np  
import scipy.io as sio  
from scipy import signal  
from tqdm import tqdm

VALID\_MAT\_PATTERN \= re.compile(r"^C\[1-8\]\_S\\d+\_M\\d+\_P\\d+\_X\\d+\_I\\d+\$")

class SlidingPreprocessor:  
    def \_\_init\_\_(  
        self,  
        raw\_dir: str,  
        processed\_dir: str,  
        audio\_sr\_orig: int \= 44100,  
        haptic\_sr\_orig: int \= 10000,  
        working\_sr: int \= 16000,  
        audio\_duration: float \= 1.0,  
        haptic\_duration: float \= 0.3,  
        lowcut: float \= 30.0,  
        highcut: float \= 800.0,  
    ):  
        self.raw\_dir \= Path(raw\_dir)  
        self.processed\_dir \= Path(processed\_dir)  
        self.audio\_sr\_orig \= audio\_sr\_orig  
        self.haptic\_sr\_orig \= haptic\_sr\_orig  
        self.working\_sr \= working\_sr  
        self.audio\_len \= int(audio\_duration \* working\_sr)   \# 16000  
        self.haptic\_len \= int(haptic\_duration \* working\_sr) \# 4800  
        self.lowcut \= lowcut  
        self.highcut \= highcut

        self.sos\_haptic \= signal.butter(  
            4, \[self.lowcut, self.highcut\], btype="bandpass", fs=self.working\_sr, output="sos"  
        )  
        self.sos\_audio\_full \= signal.butter(  
            4, \[25.0, 7600.0\], btype="bandpass", fs=self.working\_sr, output="sos"  
        )  
        self.sos\_audio\_low \= signal.butter(  
            4, \[self.lowcut, self.highcut\], btype="bandpass", fs=self.working\_sr, output="sos"  
        )  
        self.sos\_audio\_high \= signal.butter(  
            4, \[800.0, 6000.0\], btype="bandpass", fs=self.working\_sr, output="sos"  
        )

    @staticmethod  
    def \_rms\_env(x: np.ndarray, win: int) \-\> np.ndarray:  
        kernel \= np.ones(win, dtype=np.float64) / win  
        return np.sqrt(np.convolve(x \*\* 2, kernel, mode="same") \+ 1e-8)

    @staticmethod  
    def \_zscore(x: np.ndarray) \-\> np.ndarray:  
        return (x \- np.mean(x)) / (np.std(x) \+ 1e-8)

    def \_align\_audio\_window(  
        self, audio\_1s: np.ndarray, haptic\_03s: np.ndarray  
    ) \-\> tuple\[np.ndarray, np.ndarray\]:  
        """  
        Aligns the 0.3s (4800-sample) haptic vibration against the 1.0s (16000-sample) audio  
        using multi-scale (win=64 and win=256) dual-band (30-800Hz \+ 800-6000Hz) envelope  
        cross-correlation at a fine 32-sample (2ms) step.  
        """  
        audio\_low \= signal.sosfiltfilt(self.sos\_audio\_low, audio\_1s, padtype="odd", padlen=min(512, len(audio\_1s) \- 1))  
        audio\_high \= signal.sosfiltfilt(self.sos\_audio\_high, audio\_1s, padtype="odd", padlen=min(512, len(audio\_1s) \- 1))

        \# Multi-scale target envelopes  
        env\_h\_256 \= self.\_zscore(self.\_rms\_env(haptic\_03s, 256))  
        env\_h\_64 \= self.\_zscore(self.\_rms\_env(haptic\_03s, 64))

        \# Dual-band audio envelopes  
        env\_al\_256 \= self.\_rms\_env(audio\_low, 256\)  
        env\_ah\_256 \= self.\_rms\_env(audio\_high, 256\)  
        env\_al\_64 \= self.\_rms\_env(audio\_low, 64\)  
        env\_ah\_64 \= self.\_rms\_env(audio\_high, 64\)

        min\_start \= int(0.05 \* self.working\_sr)  
        max\_start \= self.audio\_len \- self.haptic\_len \- int(0.05 \* self.working\_sr)  
        stride \= 32  \# 2ms fine resolution

        best\_score \= \-10.0  
        best\_start \= int(0.35 \* self.working\_sr)

        for start in range(min\_start, max\_start \+ 1, stride):  
            end \= start \+ self.haptic\_len  
            al256 \= self.\_zscore(env\_al\_256\[start:end\])  
            ah256 \= self.\_zscore(env\_ah\_256\[start:end\])  
            al64 \= self.\_zscore(env\_al\_64\[start:end\])  
            ah64 \= self.\_zscore(env\_ah\_64\[start:end\])

            comb256 \= 0.65 \* al256 \+ 0.35 \* ah256  
            comb64 \= 0.65 \* al64 \+ 0.35 \* ah64

            c256 \= np.mean(comb256 \* env\_h\_256)  
            c64 \= np.mean(comb64 \* env\_h\_64)  
            score \= 0.6 \* c256 \+ 0.4 \* c64

            if score \> best\_score:  
                best\_score \= score  
                best\_start \= start

        aligned\_03s \= audio\_1s\[best\_start : best\_start \+ self.haptic\_len\].copy()  
        aligned\_low \= audio\_low\[best\_start : best\_start \+ self.haptic\_len\]

        \# Align waveform polarity if low-band acoustic vibration is inverted relative to accelerometer  
        if np.sum(aligned\_low \* haptic\_03s) \< 0:  
            aligned\_03s \= \-aligned\_03s  
            audio\_1s \= \-audio\_1s

        remaining \= np.concatenate(\[audio\_1s\[:best\_start\], audio\_1s\[best\_start \+ self.haptic\_len :\]\])  
        audio\_reordered \= np.concatenate(\[aligned\_03s, remaining\])\[: self.audio\_len\]  
        return audio\_reordered.astype(np.float32), haptic\_03s.astype(np.float32)

    def \_extract\_struct\_array(self, mat: dict, keys: tuple) \-\> np.ndarray \| None:  
        \# Check top-level dictionary first, then nested 'finalMaterialRecording' struct  
        for k in keys:  
            if k in mat:  
                arr \= np.asarray(mat\[k\], dtype=np.float64).squeeze()  
                if arr.ndim \== 1 and len(arr) \> 500:  
                    return arr  
        if "finalMaterialRecording" in mat:  
            rec \= mat\["finalMaterialRecording"\]  
            if isinstance(rec, np.ndarray) and rec.dtype.names is not None:  
                rec\_item \= rec\[0, 0\] if rec.ndim \== 2 else rec.item()  
                for k in keys:  
                    if k in rec.dtype.names:  
                        arr \= np.asarray(rec\_item\[k\], dtype=np.float64).squeeze()  
                        if arr.ndim \== 1 and len(arr) \> 500:  
                            return arr  
            elif isinstance(rec, dict):  
                for k in keys:  
                    if k in rec:  
                        arr \= np.asarray(rec\[k\], dtype=np.float64).squeeze()  
                        if arr.ndim \== 1 and len(arr) \> 500:  
                            return arr  
        return None

    def process\_file(self, mat\_path: Path) \-\> bool:  
        stem \= mat\_path.stem  
        if not VALID\_MAT\_PATTERN.match(stem):  
            return False

        try:  
            mat \= sio.loadmat(str(mat\_path), simplify\_cells=True)  
        except Exception:  
            return False

        raw\_audio \= self.\_extract\_struct\_array(mat, ("sound", "soundSteelTooltip"))  
        raw\_haptic \= self.\_extract\_struct\_array(mat, ("accelDFT", "accelDFTSteelTooltip"))  
        if raw\_audio is None or raw\_haptic is None:  
            return False

        \# 1\. Process Audio: Detrend \-\> Polyphase Resample (44100 \-\> 16000\) \-\> Bandpass  
        raw\_audio \= signal.detrend(raw\_audio, type="constant")  
        audio\_16k \= signal.resample\_poly(raw\_audio, up=160, down=441, padtype="line")  
        if len(audio\_16k) \>= self.audio\_len:  
            audio\_16k \= audio\_16k\[: self.audio\_len\]  
        else:  
            audio\_16k \= np.pad(audio\_16k, (0, self.audio\_len \- len(audio\_16k)), mode="reflect")

        audio\_16k \= signal.sosfiltfilt(  
            self.sos\_audio\_full, audio\_16k, padtype="odd", padlen=min(1024, len(audio\_16k) \- 1\)  
        )  
        peak\_a \= np.percentile(np.abs(audio\_16k), 99.8) \+ 1e-8  
        audio\_norm \= np.clip(audio\_16k / peak\_a, \-1.0, 1.0).astype(np.float32)

        \# 2\. Process Haptic (accelDFT): Detrend \-\> Mild Edge Taper \-\> Polyphase Resample \-\> Bandpass  
        raw\_haptic \= signal.detrend(raw\_haptic, type="linear")  
        raw\_haptic \= raw\_haptic \- np.mean(raw\_haptic)  
        raw\_haptic \= raw\_haptic \* signal.windows.tukey(len(raw\_haptic), alpha=0.04)

        haptic\_16k \= signal.resample\_poly(raw\_haptic, up=8, down=5, padtype="line")  
        if len(haptic\_16k) \>= self.haptic\_len:  
            haptic\_16k \= haptic\_16k\[: self.haptic\_len\]  
        else:  
            haptic\_16k \= np.pad(haptic\_16k, (0, self.haptic\_len \- len(haptic\_16k)), mode="reflect")

        haptic\_filtered \= signal.sosfiltfilt(  
            self.sos\_haptic, haptic\_16k, padtype="odd", padlen=min(900, len(haptic\_16k) \- 1\)  
        )  
        haptic\_filtered \= haptic\_filtered \* signal.windows.tukey(self.haptic\_len, alpha=0.03)  
        haptic\_filtered \= haptic\_filtered \- np.mean(haptic\_filtered)

        \# Strict Unit-RMS \= 1.0 Normalization with Soft Outlier Compression  
        rms\_h \= np.sqrt(np.mean(haptic\_filtered \*\* 2)) \+ 1e-8  
        haptic\_norm \= haptic\_filtered / rms\_h  
        \# Soft-compress extreme spikes (\>3.5 sigma) so they don't hit 5.0 and force RMS down  
        haptic\_norm \= 4.2 \* np.tanh(haptic\_norm / 4.2)  
        haptic\_norm \= haptic\_norm \- np.mean(haptic\_norm)  
        rms\_post \= np.sqrt(np.mean(haptic\_norm \*\* 2)) \+ 1e-8  
        haptic\_norm \= np.clip(haptic\_norm / rms\_post, \-5.0, 5.0).astype(np.float32)

        \# 3\. Align 0.3s audio window to 0.3s haptic envelope  
        audio\_aligned, haptic\_final \= self.\_align\_audio\_window(audio\_norm, haptic\_norm)

        np.save(self.processed\_dir / f"{stem}\_sliding\_audio.npy", audio\_aligned)  
        np.save(self.processed\_dir / f"{stem}\_sliding\_haptic.npy", haptic\_final)  
        return True

    def run(self):  
        self.processed\_dir.mkdir(parents=True, exist\_ok=True)  
        for stale in self.processed\_dir.glob("tmp\*"):  
            stale.unlink(missing\_ok=True)

        mat\_files \= sorted(self.raw\_dir.rglob("\*.mat"))  
        valid\_count \= 0  
        skipped\_count \= 0  
        for mat\_path in tqdm(mat\_files, desc="Preprocessing Controlled Sliding"):  
            if self.process\_file(mat\_path):  
                valid\_count \+= 1  
            else:  
                skipped\_count \+= 1  
        print(  
            f"Successfully preprocessed {valid\_count} valid sliding pairs into "  
            f"{self.processed\_dir} (skipped {skipped\_count})"  
        )  
\`\`\`

\---

\#\#\# STEP 2: Redesign \`src/models/decoder\_sliding.py\` (Acoustic-Guided Coherent Filterbank \+ Butterworth Roll-Off \+ Direct Envelope Tracker)

\#\#\#\# Why This Fixes All 4 Model Bottlenecks:  
1\. \*\*Direct Raw-Audio Conditioning (\`forward(self, latents, audio=None)\`):\*\*  
   \- Accepts \`audio\` (\`\[B, 1, 16000\]\` or \`\[B, 1, 4800\]\`) alongside EnCodec \`latents\` (\`\[B, 128, 75\]\`).  
   \- Extracts \*\*64-bin Log-Power Spectral features (\`20–4000 Hz\`)\*\* directly from \`audio\[:, :, :4800\]\` via differentiable STFT (\`\_extract\_acoustic\_spectrum\`), giving the decoder exact knowledge of the material's acoustic spectral centroid, spectral tilt, and roughness\!  
   \- Extracts \*\*Multi-Scale Acoustic Sub-Band Envelopes (\`30–800 Hz\` low-band \+ \`800–6000 Hz\` demodulated high-band)\*\* at full \`4800\`-sample resolution (\`\_extract\_acoustic\_envelopes\`) so the output envelope is directly anchored to the real physical sliding events\!  
2\. \*\*Deterministic Coherent Excitation Seed (Replaces Random Per-Step \`torch.randn\`):\*\*  
   \- Instead of calling \`torch.randn\` every forward pass (which created the STFT speckle floor that stalled training at Epoch 34), \`DifferentiableTactileFilterbank\` uses a \*\*Phase-Locked Coherent Excitation\*\* composed of:  
     \- A fixed, buffer-registered unitary broadband Schroeder/Velvet excitation basis \`base\_excitation\` \`\[1, K, 4800\]\` (deterministic across epochs so STFT gradients are 100% noise-free and smooth\!)  
     \- Plus the whitened acoustic sub-band excitation extracted directly from \`audio\[:, :, :4800\]\`\!  
3\. \*\*40-Band Filterbank Spanning \`25–1600 Hz\` with Butterworth Roll-Off:\*\*  
   \- Extends the filterbank from \`800 Hz\` up to \`1600 Hz\` with a precomputed 4th-order Butterworth roll-off curve above \`800 Hz\`, eliminating the \`-115 dB\` stopband cliff in \`evaluation\_aggregate\_psd\_2.jpg\` and slashing \`Full-Band LSD\` and \`SPQI-Lite\`\!  
4\. \*\*Explicit Multi-Scale Envelope Modulation & Crest/RMS Calibration:\*\*  
   \- Directly shapes the synthesized vibration's \`256\`-sample and \`64\`-sample temporal envelope using a learned residual gate around the acoustic sub-band envelope, boosting \`Env Corr (64, 256, 1024)\` significantly.

\#\#\#\# Complete Drop-In Code for \`src/models/decoder\_sliding.py\`:  
\`\`\`python  
import math  
import torch  
import torch.nn as nn  
import torch.nn.functional as F

class SnakeActivation(nn.Module):  
    """Periodic activation function (x \+ (1/a) \* sin^2(a \* x)) for tactile oscillation synthesis."""  
    def \_\_init\_\_(self, channels: int, init\_alpha: float \= 1.0):  
        super().\_\_init\_\_()  
        self.alpha \= nn.Parameter(torch.full((1, channels, 1), init\_alpha))

    def forward(self, x: torch.Tensor) \-\> torch.Tensor:  
        a \= self.alpha.clamp(min=0.1)  
        return x \+ (1.0 / a) \* torch.pow(torch.sin(a \* x), 2\)

class SEChannelAttention(nn.Module):  
    def \_\_init\_\_(self, channels: int, reduction: int \= 4):  
        super().\_\_init\_\_()  
        self.fc \= nn.Sequential(  
            nn.Linear(channels, channels // reduction),  
            nn.GELU(),  
            nn.Linear(channels // reduction, channels),  
            nn.Sigmoid(),  
        )

    def forward(self, x: torch.Tensor) \-\> torch.Tensor:  
        w \= x.mean(dim=-1)  
        w \= self.fc(w).unsqueeze(-1)  
        return x \* w

class DilatedResBlock(nn.Module):  
    def \_\_init\_\_(self, channels: int, kernel\_size: int \= 3, dilation: int \= 1):  
        super().\_\_init\_\_()  
        pad \= (kernel\_size \- 1\) \* dilation // 2  
        self.block \= nn.Sequential(  
            SnakeActivation(channels),  
            nn.Conv1d(channels, channels, kernel\_size, padding=pad, dilation=dilation),  
            nn.GroupNorm(8, channels),  
            SnakeActivation(channels),  
            nn.Conv1d(channels, channels, kernel\_size=1),  
        )  
        self.shortcut \= nn.Conv1d(channels, channels, kernel\_size=1)

    def forward(self, x: torch.Tensor) \-\> torch.Tensor:  
        return self.shortcut(x) \+ self.block(x)

class AntiCheckerboardUpsampleBlock(nn.Module):  
    def \_\_init\_\_(self, in\_ch: int, out\_ch: int, scale\_factor: int, kernel\_size: int \= 7):  
        super().\_\_init\_\_()  
        self.scale\_factor \= scale\_factor  
        pad \= (kernel\_size \- 1\) // 2  
        self.conv \= nn.Sequential(  
            nn.ReflectionPad1d(pad),  
            nn.Conv1d(in\_ch, out\_ch, kernel\_size=kernel\_size),  
            nn.GroupNorm(8, out\_ch),  
        )  
        self.res1 \= DilatedResBlock(out\_ch, kernel\_size=3, dilation=1)  
        self.res2 \= DilatedResBlock(out\_ch, kernel\_size=3, dilation=3)  
        self.res3 \= DilatedResBlock(out\_ch, kernel\_size=3, dilation=9)

    def forward(self, x: torch.Tensor) \-\> torch.Tensor:  
        x \= F.interpolate(x, scale\_factor=self.scale\_factor, mode="linear", align\_corners=False)  
        x \= self.conv(x)  
        x \= self.res1(x)  
        x \= self.res2(x)  
        x \= self.res3(x)  
        return x

class DifferentiableTactileFilterbank(nn.Module):  
    """  
    Coherent Multi-Band Tactile Synthesizer (40 log-spaced bands from 25 Hz to 1600 Hz).  
    \- Uses a deterministic broadband multi-sine/velvet basis \+ acoustic excitation seed instead of  
      fresh per-step torch.randn() to eliminate STFT speckle gradient noise and envelope jitter.  
    \- Includes 8 roll-off bands (800-1600 Hz) shaped with a 4th-order Butterworth envelope  
      to match the ground-truth filter roll-off and eliminate the \>800 Hz PSD cliff.  
    """  
    def \_\_init\_\_(  
        self,  
        num\_bands: int \= 40,  
        filter\_len: int \= 513,  
        target\_length: int \= 4800,  
        sr: int \= 16000,  
        f\_min: float \= 25.0,  
        f\_max: float \= 1600.0,  
    ):  
        super().\_\_init\_\_()  
        self.num\_bands \= num\_bands  
        self.filter\_len \= filter\_len  
        self.target\_length \= target\_length  
        self.sr \= sr

        freqs \= torch.logspace(math.log10(f\_min), math.log10(f\_max), num\_bands)  
        self.register\_buffer("center\_freqs", freqs)

        \# 4th-order Butterworth lowpass magnitude response at 800 Hz for natural \>800Hz roll-off  
        butter\_gain \= 1.0 / torch.sqrt(1.0 \+ torch.pow(freqs / 800.0, 8))  
        self.register\_buffer("butter\_rolloff", butter\_gain.view(1, num\_bands, 1))

        t \= torch.arange(-(filter\_len // 2), filter\_len // 2 \+ 1, dtype=torch.float32) / sr  
        window \= torch.hann\_window(filter\_len)

        filters \= \[\]  
        for i in range(num\_bands):  
            fc \= freqs\[i\].item()  
            bw \= max(fc \* 0.16, 12.0)  
            h \= 2.0 \* (bw / sr) \* torch.sinc(bw \* t) \* torch.cos(2.0 \* math.pi \* fc \* t)  
            h \= h \* window  
            h \= h / (torch.norm(h, p=2) \+ 1e-8)  
            filters.append(h)

        fir\_bank \= torch.stack(filters, dim=0).unsqueeze(1)  \# \[K, 1, filter\_len\]  
        self.register\_buffer("fir\_bank", fir\_bank)

        \# Precompute a deterministic, uniform-envelope broadband excitation basis (seeded once)  
        gen \= torch.Generator()  
        gen.manual\_seed(2026)  
        raw\_basis \= torch.randn(1, num\_bands, target\_length \+ filter\_len \- 1, generator=gen)  
        band\_basis \= F.conv1d(raw\_basis, fir\_bank, groups=num\_bands)  \# \[1, K, target\_length\]  
        \# Flatten local envelope fluctuations in the fixed basis so it has constant local RMS \= 1.0  
        local\_pwr \= F.avg\_pool1d(band\_basis.pow(2), kernel\_size=129, stride=1, padding=64)  
        band\_basis \= band\_basis / torch.sqrt(local\_pwr \+ 1e-6)  
        band\_basis \= band\_basis / (band\_basis.std(dim=-1, keepdim=True) \+ 1e-6)  
        self.register\_buffer("coherent\_basis", band\_basis)

        \# High-frequency bandpass filter (800-5500 Hz) to extract demodulated friction impulses from audio  
        t\_hf \= torch.arange(-64, 65, dtype=torch.float32) / sr  
        h\_hf \= (  
            2.0 \* (4000.0 / sr) \* torch.sinc(4000.0 \* t\_hf) \* torch.cos(2.0 \* math.pi \* 3000.0 \* t\_hf)  
            \* torch.hann\_window(129)  
        )  
        h\_hf \= h\_hf / (torch.norm(h\_hf, p=2) \+ 1e-8)  
        self.register\_buffer("hf\_filter", h\_hf.view(1, 1, 129))

    def extract\_audio\_subbands(self, audio\_03s: torch.Tensor):  
        """  
        Extracts band-decomposed acoustic excitations and multi-scale envelopes directly from raw 0.3s audio.  
        audio\_03s: \[B, 1, 4800\]  
        """  
        pad \= self.filter\_len // 2  
        a\_pad \= F.pad(audio\_03s, (pad, pad), mode="reflect")  
        \# Filter raw audio into the 40 tactile bands \[B, K, 4800\]  
        audio\_bands \= F.conv1d(a\_pad.expand(-1, self.num\_bands, \-1), self.fir\_bank, groups=self.num\_bands)  
        audio\_bands\_norm \= audio\_bands / (audio\_bands.std(dim=-1, keepdim=True) \+ 1e-5)

        \# Also extract demodulated high-frequency friction envelope (800-5000 Hz rectified)  
        a\_hf\_pad \= F.pad(audio\_03s, (64, 64), mode="reflect")  
        hf\_sig \= F.conv1d(a\_hf\_pad, self.hf\_filter).abs()  \# \[B, 1, 4800\]

        \# Low-frequency mechanical vibration energy (30-800 Hz sum)  
        lf\_sig \= audio\_bands\[:, :32, :\].pow(2).mean(dim=1, keepdim=True).sqrt()

        \# Multi-scale smooth acoustic envelope \[B, 2, 4800\]  
        env\_lf \= torch.sqrt(F.avg\_pool1d(lf\_sig.pow(2), kernel\_size=129, stride=1, padding=64) \+ 1e-8)  
        env\_hf \= torch.sqrt(F.avg\_pool1d(hf\_sig.pow(2), kernel\_size=129, stride=1, padding=64) \+ 1e-8)  
        env\_lf \= env\_lf / (env\_lf.mean(dim=-1, keepdim=True) \+ 1e-6)  
        env\_hf \= env\_hf / (env\_hf.mean(dim=-1, keepdim=True) \+ 1e-6)

        return audio\_bands\_norm, torch.cat(\[env\_lf, env\_hf\], dim=1)

    def forward(  
        self,  
        band\_gains: torch.Tensor,  
        time\_mod: torch.Tensor,  
        audio\_03s: torch.Tensor \| None \= None,  
        coherence\_mix: torch.Tensor \| None \= None,  
    ) \-\> torch.Tensor:  
        """  
        band\_gains: \[B, K, 1\] material-specific spectral band weights (learned from audio spectrum \+ latents)  
        time\_mod: \[B, K, L\] fine temporal modulation  
        """  
        B, K, L \= time\_mod.shape  
        basis \= self.coherent\_basis\[:, :, :L\].expand(B, \-1, \-1)

        if audio\_03s is not None:  
            audio\_bands\_norm, \_ \= self.extract\_audio\_subbands(audio\_03s)  
            alpha \= coherence\_mix if coherence\_mix is not None else 0.45  
            excitation \= alpha \* audio\_bands\_norm \+ torch.sqrt(1.0 \- alpha \*\* 2 \+ 1e-6) \* basis  
        else:  
            excitation \= basis

        \# Apply Butterworth roll-off above 800 Hz \+ learned per-band spectral gains \+ temporal envelope  
        effective\_gains \= band\_gains \* self.butter\_rolloff  
        shaped \= (excitation \* effective\_gains \* time\_mod).sum(dim=1, keepdim=True) / math.sqrt(K)  
        return shaped

class SlidingSEANetDecoder(nn.Module):  
    """  
    Phase 2.3 Acoustic-Guided Coherent SEANet Decoder for Controlled Sliding Synthesis.  
    Combines EnCodec latents with direct raw-audio spectral & envelope conditioning when available.  
    """  
    def \_\_init\_\_(  
        self,  
        latent\_dim: int \= 128,  
        hidden\_dim: int \= 256,  
        target\_length: int \= 4800,  
        num\_tactile\_bands: int \= 40,  
        upsample\_rates: tuple \= (5, 4, 4, 2),  
        dropout: float \= 0.1,  
    ):  
        super().\_\_init\_\_()  
        self.latent\_dim \= latent\_dim  
        self.hidden\_dim \= hidden\_dim  
        self.target\_length \= target\_length  
        self.num\_tactile\_bands \= num\_tactile\_bands

        \# 1\. Latent Processing (Preserves raw latent stats before InstanceNorm)  
        self.latent\_norm \= nn.InstanceNorm1d(latent\_dim, affine=True)  
        self.channel\_attn \= SEChannelAttention(latent\_dim, reduction=4)

        \# 2\. Direct Acoustic Spectral Encoder (64 log-power STFT bins from 0.3s audio \+ 256 latent stats)  
        self.acoustic\_spec\_bins \= 64  
        cond\_dim \= latent\_dim \* 2 \+ self.acoustic\_spec\_bins  
        self.global\_cond\_mlp \= nn.Sequential(  
            nn.Linear(cond\_dim, hidden\_dim),  
            nn.LayerNorm(hidden\_dim),  
            nn.GELU(),  
            nn.Linear(hidden\_dim, hidden\_dim),  
            nn.GELU(),  
        )  
        self.film\_proj \= nn.Linear(hidden\_dim, hidden\_dim \* 2\)

        \# Direct Spectral Band Predictor (Maps global acoustic+latent conditioning \-\> 40 filterbank band gains)  
        self.band\_gain\_predictor \= nn.Sequential(  
            nn.Linear(hidden\_dim, hidden\_dim),  
            nn.GELU(),  
            nn.Linear(hidden\_dim, num\_tactile\_bands),  
            nn.Softplus(),  
        )

        \# 3\. Sequence Modeling Backbone  
        self.input\_proj \= nn.Conv1d(latent\_dim, hidden\_dim, kernel\_size=3, padding=1)  
        self.lstm \= nn.LSTM(  
            input\_size=hidden\_dim,  
            hidden\_size=hidden\_dim // 2,  
            num\_layers=2,  
            batch\_first=True,  
            bidirectional=True,  
            dropout=dropout,  
        )  
        self.lstm\_norm \= nn.LayerNorm(hidden\_dim)

        \# 4\. Anti-Checkerboard Upsampling Backbone (30 \-\> 4800\)  
        self.upsample\_blocks \= nn.ModuleList()  
        in\_ch \= hidden\_dim  
        for rate in upsample\_rates:  
            out\_ch \= max(in\_ch // 2, 32\)  
            self.upsample\_blocks.append(AntiCheckerboardUpsampleBlock(in\_ch, out\_ch, scale\_factor=rate))  
            in\_ch \= out\_ch

        self.final\_ch \= in\_ch

        \# 5\. Synthesis Heads  
        self.filterbank \= DifferentiableTactileFilterbank(  
            num\_bands=num\_tactile\_bands, target\_length=target\_length  
        )

        \# Fine temporal band-modulation head  
        self.time\_mod\_head \= nn.Sequential(  
            SnakeActivation(self.final\_ch),  
            nn.Conv1d(self.final\_ch, num\_tactile\_bands, kernel\_size=7, padding=3),  
            nn.Softplus(),  
        )

        \# Deterministic macro-groove waveform head  
        self.det\_head \= nn.Sequential(  
            SnakeActivation(self.final\_ch),  
            nn.Conv1d(self.final\_ch, 1, kernel\_size=7, padding=3),  
        )

        \# Acoustic Envelope Fusion Gate (combines learned envelope with raw audio low/high sub-band envelopes)  
        self.env\_fusion \= nn.Sequential(  
            nn.Conv1d(self.final\_ch \+ 2, 32, kernel\_size=15, padding=7),  
            nn.GELU(),  
            nn.Conv1d(32, 1, kernel\_size=15, padding=7),  
            nn.Softplus(),  
        )

        \# Global RMS & Mix Calibration Head  
        self.calib\_head \= nn.Sequential(  
            nn.Linear(hidden\_dim, 64),  
            nn.GELU(),  
            nn.Linear(64, 3),  
            nn.Sigmoid(),  
        )

    def \_extract\_acoustic\_spectrum(self, audio\_03s: torch.Tensor \| None, batch\_size: int, device, dtype) \-\> torch.Tensor:  
        """Computes a 64-bin log-power spectrum (20-4000 Hz) directly from the 0.3s audio."""  
        if audio\_03s is None:  
            return torch.zeros(batch\_size, self.acoustic\_spec\_bins, device=device, dtype=dtype)

        y \= audio\_03s.squeeze(1)  \# \[B, 4800\]  
        win \= torch.hann\_window(512, device=device, dtype=dtype)  
        stft \= torch.stft(y, n\_fft=512, hop\_length=128, win\_length=512, window=win, return\_complex=True)  
        pwr \= stft.abs().pow(2).mean(dim=-1)  \# \[B, 257\]  
        \# Take first 64 bins (0 to \~2000 Hz at 31.25 Hz/bin)  
        log\_spec \= torch.log10(pwr\[:, 1 : self.acoustic\_spec\_bins \+ 1\] \+ 1e-7)  
        \# Normalize per-sample mean & scale so spectral tilt and centroid are prominent  
        log\_spec \= (log\_spec \- log\_spec.mean(dim=-1, keepdim=True)) / (log\_spec.std(dim=-1, keepdim=True) \+ 1e-5)  
        return log\_spec

    def forward(self, latents: torch.Tensor, audio: torch.Tensor \| None \= None) \-\> torch.Tensor:  
        """  
        latents: \[B, 128, T\] from EnCodec.  
        audio: Optional \[B, 1, 16000\] or \[B, 1, 4800\] raw input audio for direct spectral/envelope guidance.  
        """  
        B \= latents.size(0)  
        if latents.size(-1) \> 30:  
            latents \= latents\[:, :, :23\]

        audio\_03s \= None  
        if audio is not None:  
            if audio.dim() \== 2:  
                audio \= audio.unsqueeze(1)  
            audio\_03s \= audio\[:, :, : self.target\_length\]

        \# 1\. Global Conditioning from Raw Latents \+ Direct Acoustic Spectrum  
        feat\_mean \= latents.mean(dim=-1)  
        feat\_std \= latents.std(dim=-1) \+ 1e-6  
        ac\_spec \= self.\_extract\_acoustic\_spectrum(audio\_03s, B, latents.device, latents.dtype)  
        global\_cond \= self.global\_cond\_mlp(torch.cat(\[feat\_mean, feat\_std, ac\_spec\], dim=-1))  \# \[B, hidden\_dim\]

        \# 2\. Latent Sequence Processing with FiLM Conditioning  
        x \= self.latent\_norm(latents)  
        x \= self.channel\_attn(x)  
        x \= F.interpolate(x, size=30, mode="linear", align\_corners=False)  
        x \= self.input\_proj(x)

        gamma, beta \= torch.chunk(self.film\_proj(global\_cond).unsqueeze(-1), 2, dim=1)  
        x \= x \* (1.0 \+ 0.5 \* torch.tanh(gamma)) \+ 0.5 \* beta

        x\_seq \= x.transpose(1, 2\)  
        lstm\_out, \_ \= self.lstm(x\_seq)  
        x \= x \+ self.lstm\_norm(lstm\_out).transpose(1, 2\)

        for block in self.upsample\_blocks:  
            x \= block(x)

        if x.size(-1) \!= self.target\_length:  
            x \= F.interpolate(x, size=self.target\_length, mode="linear", align\_corners=False)

        \# 3\. Coherent Multi-Band Synthesis  
        band\_gains \= self.band\_gain\_predictor(global\_cond).unsqueeze(-1) \+ 0.05  \# \[B, 40, 1\]  
        time\_mod \= self.time\_mod\_head(x) \+ 0.1  
        time\_mod \= time\_mod / (time\_mod.mean(dim=-1, keepdim=True) \+ 1e-6)

        calib \= self.calib\_head(global\_cond).unsqueeze(-1)  \# \[B, 3, 1\] in (0, 1\)  
        coherence\_mix \= 0.25 \+ 0.55 \* calib\[:, 0:1, :\]       \# \[0.25, 0.80\]  
        det\_weight \= 0.35 \* calib\[:, 1:2, :\]                 \# \[0.0, 0.35\]  
        target\_rms \= 0.85 \+ 0.20 \* calib\[:, 2:3, :\]          \# \[0.85, 1.05\]

        y\_stoch \= self.filterbank(band\_gains, time\_mod, audio\_03s=audio\_03s, coherence\_mix=coherence\_mix)  
        y\_det \= self.det\_head(x)

        raw\_vib \= (1.0 \- det\_weight) \* y\_stoch \+ det\_weight \* y\_det

        \# 4\. Direct Acoustic-Guided Envelope Modulation  
        if audio\_03s is not None:  
            \_, ac\_envs \= self.filterbank.extract\_audio\_subbands(audio\_03s)  \# \[B, 2, 4800\]  
        else:  
            ac\_envs \= torch.ones(B, 2, self.target\_length, device=x.device, dtype=x.dtype)

        env\_gate \= self.env\_fusion(torch.cat(\[x, ac\_envs\], dim=1))  \# \[B, 1, 4800\]  
        env\_gate \= env\_gate / (torch.sqrt(env\_gate.pow(2).mean(dim=-1, keepdim=True)) \+ 1e-6)

        \# Normalize local carrier variance before applying the fused temporal envelope  
        carrier\_local\_rms \= torch.sqrt(F.avg\_pool1d(raw\_vib.pow(2), kernel\_size=257, stride=1, padding=128) \+ 1e-6)  
        carrier\_whitened \= raw\_vib / (0.6 \* carrier\_local\_rms \+ 0.4 \* raw\_vib.std(dim=-1, keepdim=True) \+ 1e-6)

        out \= carrier\_whitened \* env\_gate  
        out \= out \- out.mean(dim=-1, keepdim=True)

        \# Calibrate global RMS to target\_rms (\~0.95)  
        current\_rms \= torch.sqrt(out.pow(2).mean(dim=-1, keepdim=True) \+ 1e-6)  
        out \= out \* (target\_rms / current\_rms)

        return F.hardtanh(out, min\_val=-6.0, max\_val=6.0)  
\`\`\`

\---

\#\#\# STEP 3: Overhaul \`src/models/loss\_sliding.py\` (Remove Speckle-Sensitive 2D STFT Frame Loss & Optimize Welch PSD \+ Cumulative Spectral Distribution)

\#\#\#\# Why This Fixes the Epoch 34 Plateau:  
In Phase 2.2, \`\_multi\_res\_stft\_loss\` computed the Frobenius norm of the 2D spectrogram \`\|STFT\_pred(f, t) \- STFT\_gt(f, t)\|\`. On stochastic friction signals, 2D time-frequency bin differences are dominated by random speckle noise, creating a flat \`\~4.0\` loss floor.  
In Phase 2.3, we replace 2D STFT bin-by-bin difference with:  
1\. \*\*Multi-Resolution Welch PSD Loss (\`n\_fft \= 256, 512, 1024, 2048\` averaged over time frames):\*\* Time-averaging the power spectrum across frames eliminates stochastic speckle variance while supervising the exact frequency shape at 4 resolutions across both the \*\*Tactile Passband (\`30–800 Hz\`)\*\* and the \*\*Roll-off Band (\`30–2000 Hz\`)\*\*\!  
2\. \*\*Cumulative Spectral Distribution (1D Wasserstein / Earth Mover's Distance) \+ Centroid/Spread Loss:\*\*  
   \- Computes the Cumulative Distribution Function (CDF) of the normalized tactile PSD $F(f)=\sum\limits_{k\leq f}^{}{p}_{norm}(k)$ and minimizes \`F.l1\_loss(cdf\_pred, cdf\_gt)\`.  
   \- Unlike raw L1 centroid loss (which can get stuck when two peaks balance each other), the 1D Spectral Wasserstein (CDF) distance provides a convex, non-vanishing gradient that transports spectral energy directly to the exact ground-truth frequency band\!  
3\. \*\*Zero-Mean Pearson Correlation \+ L1 Multi-Scale Envelope Loss (\`win \= 64, 128, 256, 512, 1024\`):\*\*  
   \- Directly optimizes \`1.0 \- corr\` on the multi-scale RMS envelopes with higher weight on \`win \= 64\` and \`win \= 256\`.

\#\#\#\# Complete Drop-In Code for \`src/models/loss\_sliding.py\`:  
\`\`\`python  
import torch  
import torch.nn as nn  
import torch.nn.functional as F

class SlidingTextureLoss(nn.Module):  
    """  
    Phase 2.3 Speckle-Free Spectral & Perceptual Envelope Loss for Controlled Sliding.  
    Replaces noisy 2D STFT frame-bin differences with Multi-Resolution Welch PSD Loss,  
    1D Spectral Wasserstein (CDF) Distance, Differentiable Centroid/Spread Loss,  
    and Multi-Scale Envelope Correlation Loss.  
    """  
    def \_\_init\_\_(  
        self,  
        sr: int \= 16000,  
        f\_min: float \= 30.0,  
        f\_max: float \= 800.0,  
        w\_psd: float \= 4.0,  
        w\_centroid: float \= 4.0,  
        w\_env: float \= 3.5,  
        w\_stat: float \= 2.0,  
        w\_div: float \= 1.5,  
    ):  
        super().\_\_init\_\_()  
        self.sr \= sr  
        self.f\_min \= f\_min  
        self.f\_max \= f\_max  
        self.w\_psd \= w\_psd  
        self.w\_centroid \= w\_centroid  
        self.w\_env \= w\_env  
        self.w\_stat \= w\_stat  
        self.w\_div \= w\_div  
        self.fft\_sizes \= (256, 512, 1024, 2048\)  
        self.env\_windows \= (64, 128, 256, 512, 1024\)

    def \_compute\_welch\_psd\_and\_moments(self, y: torch.Tensor, n\_fft: int \= 1024):  
        hop \= n\_fft // 4  
        win \= torch.hann\_window(n\_fft, device=y.device, dtype=y.dtype)  
        stft \= torch.stft(y, n\_fft=n\_fft, hop\_length=hop, win\_length=n\_fft, window=win, return\_complex=True)  
        power \= stft.abs().pow(2).mean(dim=-1)  \# Welch PSD \[B, F\]

        freqs \= torch.linspace(0, self.sr / 2.0, power.size(-1), device=y.device, dtype=y.dtype)  
        mask\_tac \= (freqs \>= self.f\_min) & (freqs \<= self.f\_max)  
        mask\_wide \= (freqs \>= 20.0) & (freqs \<= 2000.0)

        p\_tac \= power\[:, mask\_tac\] \+ 1e-8  
        f\_tac \= freqs\[mask\_tac\].unsqueeze(0)

        p\_norm \= p\_tac / (p\_tac.sum(dim=-1, keepdim=True) \+ 1e-8)  
        cdf \= torch.cumsum(p\_norm, dim=-1)

        centroid \= (p\_norm \* f\_tac).sum(dim=-1)  
        spread \= torch.sqrt(((f\_tac \- centroid.unsqueeze(-1)).pow(2) \* p\_norm).sum(dim=-1) \+ 1e-6)

        log\_p\_tac \= 10.0 \* torch.log10(p\_tac)  
        max\_tac \= log\_p\_tac.max(dim=-1, keepdim=True)\[0\]  
        log\_p\_tac \= torch.maximum(log\_p\_tac, max\_tac \- 55.0)

        p\_wide \= power\[:, mask\_wide\] \+ 1e-9  
        log\_p\_wide \= 10.0 \* torch.log10(p\_wide)  
        max\_wide \= log\_p\_wide.max(dim=-1, keepdim=True)\[0\]  
        log\_p\_wide \= torch.maximum(log\_p\_wide, max\_wide \- 60.0)

        return p\_tac, log\_p\_tac, log\_p\_wide, p\_norm, cdf, centroid, spread

    def \_multi\_res\_welch\_loss(self, pred: torch.Tensor, target: torch.Tensor):  
        total\_psd\_loss \= 0.0  
        total\_moment\_loss \= 0.0

        for n\_fft in self.fft\_sizes:  
            p\_tac, log\_p\_tac, log\_p\_wide, p\_norm, cdf\_p, cent\_p, spread\_p \= self.\_compute\_welch\_psd\_and\_moments(pred, n\_fft)  
            t\_tac, log\_t\_tac, log\_t\_wide, t\_norm, cdf\_t, cent\_t, spread\_t \= self.\_compute\_welch\_psd\_and\_moments(target, n\_fft)

            \# Passband LSD \+ Wideband Roll-off LSD (prevents \>800Hz cliff)  
            lsd\_tac \= torch.sqrt(F.mse\_loss(log\_p\_tac, log\_t\_tac) \+ 1e-6) / 10.0  
            lsd\_wide \= torch.sqrt(F.mse\_loss(log\_p\_wide, log\_t\_wide) \+ 1e-6) / 15.0

            cos\_lin \= 1.0 \- F.cosine\_similarity(p\_tac, t\_tac, dim=-1).mean()  
            cos\_log \= 1.0 \- F.cosine\_similarity(log\_p\_tac, log\_t\_tac, dim=-1).mean()

            total\_psd\_loss \= total\_psd\_loss \+ lsd\_tac \+ 0.4 \* lsd\_wide \+ cos\_lin \+ 0.5 \* cos\_log

            \# 1D Wasserstein (CDF) Distance \+ Centroid \+ Spread Loss  
            emd\_loss \= F.l1\_loss(cdf\_p, cdf\_t) \* 2.5  
            cent\_loss \= F.l1\_loss(cent\_p / self.f\_max, cent\_t / self.f\_max)  
            spr\_loss \= F.l1\_loss(spread\_p / self.f\_max, spread\_t / self.f\_max)  
            total\_moment\_loss \= total\_moment\_loss \+ emd\_loss \+ cent\_loss \+ 0.5 \* spr\_loss

        n \= len(self.fft\_sizes)  
        return total\_psd\_loss / n, total\_moment\_loss / n

    def \_envelope\_and\_stats\_loss(self, pred: torch.Tensor, target: torch.Tensor):  
        env\_loss \= 0.0  
        for w in self.env\_windows:  
            stride \= max(w // 4, 8\)  
            env\_p \= torch.sqrt(F.avg\_pool1d(pred.pow(2), kernel\_size=w, stride=stride) \+ 1e-8)  
            env\_t \= torch.sqrt(F.avg\_pool1d(target.pow(2), kernel\_size=w, stride=stride) \+ 1e-8)

            l1\_env \= F.l1\_loss(env\_p, env\_t)  
            vp \= env\_p \- env\_p.mean(dim=-1, keepdim=True)  
            vt \= env\_t \- env\_t.mean(dim=-1, keepdim=True)  
            corr \= (vp \* vt).sum(dim=-1) / (  
                torch.sqrt((vp.pow(2).sum(dim=-1) \+ 1e-8) \* (vt.pow(2).sum(dim=-1) \+ 1e-8))  
            )  
            env\_loss \= env\_loss \+ l1\_env \+ (1.0 \- corr.mean())  
        env\_loss \= env\_loss / len(self.env\_windows)

        rms\_p \= torch.sqrt(pred.pow(2).mean(dim=-1) \+ 1e-8)  
        rms\_t \= torch.sqrt(target.pow(2).mean(dim=-1) \+ 1e-8)  
        rms\_loss \= F.l1\_loss(rms\_p, rms\_t)

        peak99\_p \= torch.quantile(pred.abs(), 0.995, dim=-1)  
        peak99\_t \= torch.quantile(target.abs(), 0.995, dim=-1)  
        peak\_max\_p \= pred.abs().amax(dim=-1)  
        peak\_max\_t \= target.abs().amax(dim=-1)  
        peak\_loss \= F.l1\_loss(peak99\_p, peak99\_t) \+ 0.5 \* F.l1\_loss(peak\_max\_p, peak\_max\_t)

        crest\_p \= peak99\_p / (rms\_p \+ 1e-6)  
        crest\_t \= peak99\_t / (rms\_t \+ 1e-6)  
        crest\_loss \= F.l1\_loss(crest\_p, crest\_t)

        stat\_loss \= rms\_loss \+ peak\_loss \+ 0.5 \* crest\_loss  
        return env\_loss, stat\_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor, return\_components: bool \= False):  
        if pred.dim() \== 2:  
            pred \= pred.unsqueeze(1)  
        if target.dim() \== 2:  
            target \= target.unsqueeze(1)

        p\_flat \= pred.squeeze(1)  
        t\_flat \= target.squeeze(1)

        psd\_loss, centroid\_loss \= self.\_multi\_res\_welch\_loss(p\_flat, t\_flat)  
        env\_loss, stat\_loss \= self.\_envelope\_and\_stats\_loss(pred, target)

        \# Batch Spectral Diversity Loss at n\_fft=1024  
        if p\_flat.size(0) \> 1:  
            \_, \_, \_, p\_norm, \_, cent\_p, \_ \= self.\_compute\_welch\_psd\_and\_moments(p\_flat, n\_fft=1024)  
            \_, \_, \_, t\_norm, \_, cent\_t, \_ \= self.\_compute\_welch\_psd\_and\_moments(t\_flat, n\_fft=1024)  
            dist\_p \= torch.cdist(p\_norm, p\_norm, p=1)  
            dist\_t \= torch.cdist(t\_norm, t\_norm, p=1)  
            div\_loss \= F.l1\_loss(dist\_p, dist\_t) \+ F.l1\_loss(  
                cent\_p.std() / self.f\_max, cent\_t.std() / self.f\_max  
            )  
        else:  
            div\_loss \= torch.tensor(0.0, device=pred.device)

        total\_loss \= (  
            self.w\_psd \* psd\_loss  
            \+ self.w\_centroid \* centroid\_loss  
            \+ self.w\_env \* env\_loss  
            \+ self.w\_stat \* stat\_loss  
            \+ self.w\_div \* div\_loss  
        )

        if return\_components:  
            return total\_loss, {  
                "psd": psd\_loss.item(),  
                "cent": centroid\_loss.item(),  
                "env": env\_loss.item(),  
                "stat": stat\_loss.item(),  
                "div": div\_loss.item(),  
            }  
        return total\_loss  
\`\`\`

\---

\#\#\# STEP 4: Pass \`audio\` to \`decoder(latents, audio)\` in \`scripts/train\_sliding.py\`, \`scripts/evaluate\_sliding.py\`, and \`scripts/inference\_sliding.py\`

\#\#\#\# Changes Required in All 3 Runner Scripts:  
Inspect \`scripts/train\_sliding.py\`, \`scripts/evaluate\_sliding.py\`, and \`scripts/inference\_sliding.py\`. Wherever the decoder is called:  
\`\`\`python  
\# Previously:  
pred\_haptic \= decoder(latents)

\# Change to:  
pred\_haptic \= decoder(latents, audio=audio)  
\`\`\`  
Also in \`scripts/train\_sliding.py\`:  
\- Increase \`patience\` from \`25\` to \`40\` epochs so CosineAnnealingLR can fine-tune the filterbank and envelope gates cleanly.  
\- If \`train\_sliding.py\` unpacks loss components for the tqdm progress bar, use \`loss, comps \= criterion(pred\_haptic, haptic, return\_components=True)\` (or keep backward compatibility if \`criterion\` returns a tuple/tensor).

\---

\#\#\# STEP 5: Run Full Pipeline & Verify Target Benchmarks

Execute in terminal:  
\`\`\`bash  
uv run python scripts/run\_preprocessing\_sliding.py  
uv run python scripts/train\_sliding.py  
uv run python scripts/evaluate\_sliding.py  
uv run python scripts/inference\_sliding.py  
\`\`\`

\#\#\#\# Expected Phase 2.3 Improvements Over Phase 2.2:  
\- \*\*Training Convergence:\*\* Smoothly decreases from Epoch 1 through Epoch 80+ without stalling at Epoch 34\.  
\- \*\*\`Envelope Corr (64 / 256 / 1024)\`:\*\* Jumps from \`0.277 / 0.473 / 0.617\` to \*\*\`0.55+ / 0.72+ / 0.85+\`\*\* thanks to direct acoustic sub-band envelope fusion and coherent excitation.  
\- \*\*\`Spectral Centroid Delta (Hz)\`:\*\* Drops from \`96.06 Hz\` to \*\*\`\< 50 Hz\`\*\* via direct 64-bin acoustic STFT conditioning and 1D Spectral Wasserstein (CDF) loss.  
\- \*\*\`Tactile Band LSD\` & \`Full-Band LSD\`:\*\* \`Tactile LSD\` improves from \`3.46 dB\` to \*\*\`\< 2.6 dB\`\*\*, and \`Full-Band LSD\` drops from \`5.58 dB\` to \*\*\`\< 3.2 dB\`\*\* (eliminating the \`\>800 Hz\` PSD cliff).  
\- \*\*\`PSD Cosine Similarity\`:\*\* Improves from \`0.653\` to \*\*\`\> 0.78\`\*\*.  
\- \*\*All \`rms\_gt\` values:\*\* Strictly \`1.000\` across all 200 validation samples.

\---

\#\# 3\. Updated \`context.md\` (Replace Entire File)

\`\`\`markdown  
\# Project Context — Audio-to-Haptic Cross-Modal Synthesis

\> \*\*Last Updated:\*\* 2026-11-01 18:10 IST    
\> \*\*Project:\*\* BTP — Understanding Multisensory Information to Build Perception of Object Properties    
\> \*\*Author:\*\* ParthKhiriya (b23ee1051@iitj.ac.in)    
\> \*\*Current Task:\*\* Phase 2.3 — Acoustic-Guided Coherent Sliding Synthesis Overhaul

\---

\#\# 1\. Project Overview

This BTP investigates cross-modal relationships between vision, sound, and haptics during interactions with textured surfaces. The immediate goal is building a generative model that synthesizes perceptually meaningful haptic (vibrotactile) signals directly from audio input.

\#\#\# Research Objectives  
1\. \*\*Cross-Modal Learning\*\* — Predict haptic signals from audio (and later visual) modalities  
2\. \*\*Multimodal Interaction\*\* — Investigate relative weighting of modalities during texture interaction  
3\. \*\*Physical Features\*\* — Study perceptually validated features (hardness, roughness)  
4\. \*\*Perceptual Dimensions\*\* — Characterize perceptual space of textured surfaces during tool-tip interactions

\---

\#\# 2\. Literature Review

\| Paper \| Architecture \| Task \| Key Details \|  
\|---\|---\|---\|---\|  
\| \*\*Sound2Hap\*\* (Li & Seifi, CHI '26) \| CNN autoencoder (EnCodec encoder \+ SEANet decoder) \| Audio $\rightarrow$ vibrotactile \| Frozen EnCodec encoder, SEANet decoder with LSTM \+ residual upsampling, LeakyReLU, no tanh output. Trained with human perceptual ratings. \|  
\| \*\*HapticGen\*\* (Sung et al., CHI '25) \| Autoregressive transformer \+ EnCodec \| Text $\rightarrow$ vibrotactile \| Leverages EnCodec neural audio codec \|  
\| \*\*VibNet\*\* \| Deep learning framework \| Predict roughness/emotion from haptic \| Biomimetic approach \|  
\| \*\*Cluster Haptic Texture Dataset\*\* (Eguchi, 2026\) \| — \| Dataset: 118 surfaces \| 3-axis machine, controlled velocity/direction \|  
\| \*\*Devillard's Dataset\*\* (Imperial, 2024-25) \| — \| Dataset: 10 surfaces \| Bare-finger, multimodal \|  
\| \*\*Haptipedia\*\* (Agrawal et al.) \| — \| Device corpus \| Taxonomy for haptic device design \|

\---

\#\# 3\. Dataset — LMT Haptic Texture Database (TUM CBM)

\#\#\# 3.1 Taxonomy and File Naming Structure  
Every \`.mat\` file in \`data/LMT\_FinalDatabase/CBM\_FinalDatabase/\` follows the standard format:    
\`C\<class\>\_S\<subclass\>\_M\<material\>\_P\<processing\>\_X\<composition\>\_I\<instance\>.mat\`

\- \*\*C (Class 1–8):\*\* C1 Wood, C2 Paper/Fabric, C3 Polymers/Plastics, C4 Ceramics/Foam, C5 Glasses/Carpet/Leather, C6 Stones, C7 Metals, C8 Composites.  
\- \*\*I (Instance 1–5):\*\* 5 repeated independent measurement passes per surface.  
\- \*\*Total:\*\* 1,115 raw files; 1,000 valid paired recordings (800 train / 200 stratified validation).

\#\#\# 3.2 Modality & Interaction Decomposition  
1\. \*\*Controlled Tap Interaction (Impact Contact):\*\* \`soundTap\` (5.0s @ 44.1 kHz) $\rightarrow$ \`accelTap\` (1.5s @ 10 kHz).  
2\. \*\*Continuous Controlled Sliding (Surface Texture Friction):\*\* \`sound\` / \`soundSteelTooltip\` (1.0s @ 44.1 kHz) $\rightarrow$ \`accelDFT\` / \`accelDFTSteelTooltip\` (0.3s @ 10 kHz, 1D via \`DFT321\`).

\---

\#\# 4\. Phase 1: Controlled Tap Synthesis (\`controlled-tap\` Branch — Finalized Baseline)

\#\#\# 4.1 Quantitative Summary ($N=186$ Validation Samples)  
\- \*\*Envelope Correlation:\*\* \`Env Corr (64)\`: \*\*0.9275\*\*, \`Env Corr (256)\`: \*\*0.9577\*\*, \`Env Corr (1024)\`: \*\*0.9705\*\*  
\- \*\*Amplitude & Energy:\*\* \`Peak Ratio\`: \*\*1.1688\*\*, \`RMS Ratio\`: \*\*0.8264\*\*, \`Tactile Band Energy Ratio (50–300 Hz)\`: \*\*0.8496\*\*  
\- \*\*Spectral Fidelity:\*\* \`Tactile Band LSD\`: \*\*3.45 dB\*\* (median \*\*2.48 dB\*\*), \`PSD CosSim\`: \*\*0.8100\*\*, \`Centroid Delta\`: \*\*36.95 Hz\*\*  
\- \*\*Time Domain:\*\* \`PCC\`: \*\*0.2327\*\*, \`RMSE\`: \*\*0.5680\*\*, \`SNR\`: \*\*-0.52 dB\*\*

\---

\#\# 5\. Phase 2: Controlled Sliding Synthesis (\`controlled-sliding\` Branch)

\#\#\# 5.1 Progression of Sliding Evaluation Metrics ($N=200$ Stratified Validation Samples)

\| Metric \| Phase 2.1 (Baseline) \| Phase 2.2 (Stochastic Filterbank) \| Target (Phase 2.3) \|  
\|---\|---\|---\|---\|  
\| \*\*Peak Amplitude Ratio\*\* \| $0.4685\pm 0.1466$ \| \*\*$1.0050\pm 0.1925$\*\* (med $1.000$) \| $0.95--1.05$ \|  
\| \*\*RMS Energy Ratio\*\* \| $0.6968\pm 0.0692$ \| \*\*$0.9972\pm 0.0978$\*\* (med $0.987$) \| $0.95--1.05$ \|  
\| \*\*Tactile Band Energy Ratio\*\* \| $0.6627\pm 1.0295$ \| \*\*$0.8412\pm 0.2012$\*\* (med $0.818$) \| $0.85--1.00$ \|  
\| \*\*Tactile Band (30–800Hz) LSD\*\* \| $4.9373\pm 0.8465dB$ \| \*\*$3.4559\pm 1.0610dB$\*\* (med $3.23dB$) \| $<2.6dB$ \|  
\| \*\*Full-Band LSD\*\* \| $12.4792\pm 13.2545dB$ \| \*\*$5.5808\pm 1.6476dB$\*\* \| $<3.2dB$ \|  
\| \*\*PSD Cosine Similarity\*\* \| $0.5456\pm 0.1345$ \| \*\*$0.6533\pm 0.1274$\*\* (med $0.681$) \| $>0.78$ \|  
\| \*\*Spectral Centroid Delta\*\* \| $115.78\pm 89.95Hz$ \| \*\*$96.06\pm 62.60Hz$\*\* (med $89.0Hz$) \| $<50Hz$ \|  
\| \*\*Envelope Corr (64 / 256 / 1024)\*\* \| $0.387/0.530/0.620$ \| $0.277/0.473/0.617$ \| $>0.55/0.72/0.85$ \|  
\| \*\*SPQI-Lite\*\* \| $6.1075\pm 1.1457$ \| \*\*$5.8699\pm 1.0406$\*\* \| $<4.2$ \|

\#\#\# 5.2 Phase 2.2 Assessment & Diagnosis of Epoch 34 Early Stopping  
1\. \*\*What Phase 2.2 Solved:\*\*  
   \- Restored unit RMS (\`0.997x\`) and Peak Amplitude Ratio (\`1.005x\`).  
   \- Eliminated \`ConvTranspose1d\` high-frequency comb notches (\`evaluation\_aggregate\_psd.png\`).  
   \- Improved Tactile LSD by \$30\\%\$ (\`4.94 dB\` $\rightarrow$ \`3.46 dB\`) and eliminated boundary filter spikes at $t=0.0s$ and $t=0.3s$.  
2\. \*\*Why Phase 2.2 Plateaued at Epoch 34 (\`Val Loss: 10.0467\`):\*\*  
   \- \*\*Per-Step \`torch.randn\` STFT Speckle Noise Floor:\*\* Sampling fresh Gaussian noise on every forward pass created random frame-by-frame STFT bin variance (\`\_multi\_res\_stft\_loss\`) and random short-window envelope bumps (\`Env Corr (64)\` dropped to \`0.277\`).  
   \- \*\*EnCodec Friction Blindness:\*\* Relying solely on frozen EnCodec latents without raw-audio spectral conditioning restricted predicted centroids to \`310–515 Hz\` (\`Centroid Delta \= 96.06 Hz\`).  
   \- \*\*\`\>800 Hz\` Filterbank Cliff:\*\* Restricting filterbank bands to $\leq 800Hz$ created a \`-115 dB/Hz\` cliff between \`800–2000 Hz\`, inflating \`Full-Band LSD\` (\`5.58 dB\`).

\#\#\# 5.3 Phase 2.3 Architecture Overhaul (Current Implementation)  
1\. \*\*Dual-Band Physics Alignment & Strict Unit-RMS (\`src/data/preprocess\_sliding.py\`):\*\*  
   \- Aligns the \`0.3s\` audio window using a multi-scale (\`win=64, 256\`) combination of the \`30–800 Hz\` mechanical sub-band envelope and \`800–6000 Hz\` friction envelope at \`2 ms\` (\`32\`-sample) resolution.  
   \- Enforces strict \`rms\_gt \== 1.000\` after soft \`tanh\` outlier compression.  
2\. \*\*Acoustic-Guided Coherent Decoder (\`src/models/decoder\_sliding.py\`):\*\*  
   \- Passes \`audio\` alongside \`latents\` into \`SlidingSEANetDecoder.forward(latents, audio=audio)\`.  
   \- Extracts a 64-bin log-power spectrum directly from \`audio\[:, :, :4800\]\` to condition the 40-band filterbank (\`25–1600 Hz\` with 4th-order Butterworth roll-off above \`800 Hz\`).  
   \- Replaces per-step \`torch.randn\` with a deterministic coherent basis \+ whitened acoustic sub-band seed, and fuses the learned temporal envelope with the raw audio sub-band envelopes.  
3\. \*\*Speckle-Free Multi-Resolution Welch & Wasserstein CDF Loss (\`src/models/loss\_sliding.py\`):\*\*  
   \- Replaces 2D frame-by-frame STFT magnitude differences with Multi-Resolution Welch PSD Loss (\`256, 512, 1024, 2048\`), 1D Spectral Wasserstein (CDF) Distance, and Multi-Scale Envelope Correlation Loss.

\---

\#\# 6\. Execution & Implementation Status

\- \[x\] Controlled Tap Synthesis Phase (\`controlled-tap\`): Validated & Finalized  
\- \[x\] Phase 2.1 Initial Sliding Pipeline Implemented & Evaluated ($N=201$ samples)  
\- \[x\] Phase 2.2 Overhaul Implemented & Evaluated ($N=200$ samples; solved amplitude & comb notches; identified \`torch.randn\` speckle plateau)  
\- \[ \] Phase 2.3 Acoustic-Guided Coherent Overhaul Execution:  
  \- \[ \] Update \`src/data/preprocess\_sliding.py\` and re-run \`scripts/run\_preprocessing\_sliding.py\`  
  \- \[ \] Update \`src/models/decoder\_sliding.py\` with \`audio\`-conditioned coherent filterbank (\`25–1600 Hz\`)  
  \- \[ \] Update \`src/models/loss\_sliding.py\` with Multi-Resolution Welch PSD \+ 1D Wasserstein CDF loss  
  \- \[ \] Pass \`audio\` to \`decoder(latents, audio=audio)\` in \`train\_sliding.py\`, \`evaluate\_sliding.py\`, and \`inference\_sliding.py\`  
  \- \[ \] Retrain model and evaluate Phase 2.3 metrics  
\`\`\`  

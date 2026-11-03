import numpy as np
import torch
from scipy.signal import hilbert


def dft321_projection(acc_xyz: np.ndarray) -> np.ndarray:
    """
    Phase-preserving 3-axis (N, 3) to 1-axis (N,) accelerometer projection.
    Preserves total 3D spectral energy across X, Y, Z while locking phase to the
    principal normal impact axis (preventing frequency-doubling rectification).
    """
    acc_xyz = np.asarray(acc_xyz, dtype=np.float64)
    if acc_xyz.ndim == 1:
        return (acc_xyz - np.mean(acc_xyz)).astype(np.float32)
    if acc_xyz.shape[0] == 3 and acc_xyz.shape[1] != 3:
        acc_xyz = acc_xyz.T  # Ensure shape (N, 3)

    # Zero-mean per axis
    acc_xyz = acc_xyz - np.mean(acc_xyz, axis=0, keepdims=True)
    N = acc_xyz.shape[0]

    # FFT along time axis
    X = np.fft.rfft(acc_xyz, axis=0)  # (N_freqs, 3)
    mag_3d = np.sqrt(np.sum(np.abs(X) ** 2, axis=1))

    # Principal impact direction in time domain via covariance eigenvector
    cov = np.cov(acc_xyz.T)
    _, eigvecs = np.linalg.eigh(cov)
    principal_dir = eigvecs[:, -1]
    # Lock sign to normal Z-axis (index 2) so initial impact spike polarity is consistent
    if principal_dir[2] < 0:
        principal_dir = -principal_dir

    proj_time = acc_xyz @ principal_dir
    phase_ref = np.angle(np.fft.rfft(proj_time, n=N))

    Y_1d = mag_3d * np.exp(1j * phase_ref)
    y_1d = np.fft.irfft(Y_1d, n=N)
    return y_1d.astype(np.float32)


def compute_analytic_envelope(signal: np.ndarray, fs: int, smooth_ms: float = 0.5) -> np.ndarray:
    """Computes smoothed Hilbert analytic amplitude envelope."""
    sig = np.asarray(signal, dtype=np.float32)
    sig = sig - np.mean(sig)
    env = np.abs(hilbert(sig))
    win = max(1, int(round(smooth_ms * 0.001 * fs)))
    if win > 1:
        kernel = np.ones(win, dtype=np.float32) / float(win)
        env = np.convolve(env, kernel, mode="same")
    return env


def detect_onset_index(
    signal: np.ndarray,
    fs: int,
    threshold_ratio: float = 0.06,
    search_back_ms: float = 15.0,
) -> int:
    """
    Detects the exact first-contact foot of the primary tap transient using the
    smoothed Hilbert envelope and backward foot-walk from the peak impact.
    """
    env = compute_analytic_envelope(signal, fs, smooth_ms=0.4)
    # Estimate noise floor from first 5% of recording
    n_noise = max(16, int(0.05 * len(env)))
    noise_floor = np.median(env[:n_noise]) + 1e-7

    peak_idx = int(np.argmax(env))
    peak_val = float(env[peak_idx])
    thresh = max(noise_floor * 3.0, threshold_ratio * peak_val)

    max_back = int(round(search_back_ms * 0.001 * fs))
    start_search = max(0, peak_idx - max_back)

    onset_idx = peak_idx
    for i in range(peak_idx, start_search, -1):
        if env[i] <= thresh:
            onset_idx = i
            break
    return onset_idx


def gcc_phat_envelope_lag(
    env_ref: np.ndarray,
    env_sig: np.ndarray,
    max_shift_samples: int,
) -> int:
    """
    Finds the integer sample lag of `env_sig` relative to `env_ref` within
    [-max_shift_samples, +max_shift_samples] using GCC-PHAT on positive onset derivatives.
    """
    d_ref = np.maximum(0.0, np.diff(env_ref, prepend=env_ref[0]))
    d_sig = np.maximum(0.0, np.diff(env_sig, prepend=env_sig[0]))

    n = len(d_ref) + len(d_sig)
    n_fft = 1 << (n - 1).bit_length()
    R = np.fft.rfft(d_sig, n=n_fft) * np.conj(np.fft.rfft(d_ref, n=n_fft))
    # Blended PHAT weighting (0.75) for robustness on impulsive envelopes
    cc = np.fft.irfft(R / (np.abs(R) ** 0.75 + 1e-8), n=n_fft)

    max_shift = min(max_shift_samples, n_fft // 2 - 1)
    if max_shift <= 0:
        return 0
    cc_valid = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = int(np.argmax(cc_valid)) - max_shift
    return shift


def align_audio_and_haptic_tap(
    audio: np.ndarray,
    haptic: np.ndarray,
    fs_audio: int,
    fs_haptic: int,
    pre_ms: float = 15.0,
    post_ms: float = 113.0,
    max_align_shift_ms: float = 5.0,
) -> dict[str, np.ndarray | float]:
    """
    1. Projects 3-axis haptic acceleration to 1D phase-locked signal if needed.
    2. Detects exact tap onset in haptic and audio signals.
    3. Refines micro-alignment via envelope GCC-PHAT within [-max_align_shift_ms, +max_align_shift_ms].
    4. Extracts synchronized tap-centered windows of duration (pre_ms + post_ms).
    5. Computes decoupled log-peak amplitude for explicit gain conditioning.
    """
    if haptic.ndim > 1:
        haptic = dft321_projection(haptic)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1).astype(np.float32)

    haptic = haptic.astype(np.float32) - np.mean(haptic)
    audio = audio.astype(np.float32) - np.mean(audio)

    h_onset = detect_onset_index(haptic, fs_haptic, threshold_ratio=0.06)
    a_onset_expected = int(round(h_onset * (fs_audio / float(fs_haptic))))
    a_onset_detected = detect_onset_index(audio, fs_audio, threshold_ratio=0.06)

    max_shift_a = int(round(max_align_shift_ms * 0.001 * fs_audio))
    if abs(a_onset_detected - a_onset_expected) <= max_shift_a:
        a_onset = a_onset_detected
    else:
        a_onset = a_onset_expected

    def _crop_around(sig: np.ndarray, center: int, fs: int) -> np.ndarray:
        n_pre = int(round(pre_ms * 0.001 * fs))
        n_post = int(round(post_ms * 0.001 * fs))
        total_len = n_pre + n_post
        start = center - n_pre
        end = center + n_post
        out = np.zeros(total_len, dtype=np.float32)
        src_s = max(0, start)
        src_e = min(len(sig), end)
        dst_s = max(0, -start)
        dst_e = dst_s + (src_e - src_s)
        if src_e > src_s:
            out[dst_s:dst_e] = sig[src_s:src_e]
        return out

    h_crop = _crop_around(haptic, h_onset, fs_haptic)
    a_crop = _crop_around(audio, a_onset, fs_audio)

    # Fine GCC-PHAT alignment if sample rates match (or after resampling envelope)
    if fs_audio == fs_haptic:
        env_h = compute_analytic_envelope(h_crop, fs_haptic, smooth_ms=0.4)
        env_a = compute_analytic_envelope(a_crop, fs_audio, smooth_ms=0.4)
        fine_shift = gcc_phat_envelope_lag(env_h, env_a, max_shift_samples=int(0.002 * fs_audio))
        if abs(fine_shift) > 0:
            a_crop = _crop_around(audio, a_onset + fine_shift, fs_audio)

    peak_h = float(np.max(np.abs(h_crop)) + 1e-6)
    log_peak_gain = float(np.log(peak_h))
    h_norm = (h_crop / peak_h).astype(np.float32)

    return {
        "audio": a_crop,
        "haptic": h_crop,
        "haptic_norm": h_norm,
        "log_peak_gain": log_peak_gain,
        "peak_gain": peak_h,
    }

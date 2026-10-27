import numpy as np
from scipy import signal
from scipy.stats import pearsonr
import torch

def _to_numpy(x) -> np.ndarray:
    """Ensure input is a 1D float32 numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().squeeze().numpy()
    elif isinstance(x, np.ndarray):
        x = x.squeeze()
    return np.asarray(x, dtype=np.float32).flatten()

def compute_rms_envelope(x: np.ndarray, frame_len: int = 256, hop_len: int = 128) -> np.ndarray:
    """Compute moving RMS envelope."""
    if len(x) < frame_len:
        return np.array([np.sqrt(np.mean(x**2) + 1e-10)], dtype=np.float32)
    # Fast windowed RMS using uniform filter or strided rolling window
    pad_len = frame_len // 2
    x_padded = np.pad(x, (pad_len, pad_len), mode='reflect')
    # Use convolution of squared signal
    window = np.ones(frame_len, dtype=np.float32) / frame_len
    env = np.sqrt(np.maximum(0, np.convolve(x_padded**2, window, mode='valid')) + 1e-10)
    return env[::hop_len]

def tactile_sensitivity_weighting(freqs: np.ndarray) -> np.ndarray:
    """
    Psychophysical frequency weighting W(f) for vibrotactile perception.
    Peaks around 200-250 Hz (Pacinian channel maximum sensitivity)
    and drops towards low (<20 Hz) and high (>500 Hz) frequencies.
    """
    freqs = np.asarray(freqs, dtype=np.float32)
    w = np.zeros_like(freqs)
    valid = freqs > 5.0
    # Log-Gaussian sensitivity curve centered at 220 Hz with bandwidth matching human tuning
    center_f = 220.0
    sigma_log = np.log(2.3)
    w[valid] = np.exp(-((np.log(freqs[valid]) - np.log(center_f)) ** 2) / (2.0 * sigma_log ** 2))
    w = w / (np.max(w) + 1e-8)
    return w

def compute_time_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Computes time-domain metrics comparing prediction to ground truth."""
    pred = _to_numpy(pred)
    gt = _to_numpy(gt)
    
    min_len = min(len(pred), len(gt))
    pred = pred[:min_len]
    gt = gt[:min_len]
    
    # 1. Pearson Correlation Coefficient (PCC)
    if np.std(pred) < 1e-7 or np.std(gt) < 1e-7:
        pcc = 0.0
    else:
        r, _ = pearsonr(pred, gt)
        pcc = float(r) if not np.isnan(r) else 0.0
        
    # 2. Root Mean Squared Error (RMSE)
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    
    # 3. Signal to Noise Ratio (SNR in dB)
    noise_power = np.sum((gt - pred) ** 2)
    signal_power = np.sum(gt ** 2)
    if noise_power < 1e-10:
        snr_db = 100.0
    else:
        snr_db = float(10.0 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10)))
        
    # 4. Peak Amplitude Ratio (pred_peak / gt_peak)
    peak_pred = float(np.max(np.abs(pred)))
    peak_gt = float(np.max(np.abs(gt)))
    peak_ratio = float(peak_pred / (peak_gt + 1e-8))
    
    # 5. RMS Ratio
    rms_pred = float(np.sqrt(np.mean(pred ** 2)))
    rms_gt = float(np.sqrt(np.mean(gt ** 2)))
    rms_ratio = float(rms_pred / (rms_gt + 1e-8))
    
    # 6. Multi-scale Envelope Correlation
    env_corrs = {}
    for fl in [64, 256, 1024]:
        hl = max(1, fl // 2)
        env_pred = compute_rms_envelope(pred, frame_len=fl, hop_len=hl)
        env_gt = compute_rms_envelope(gt, frame_len=fl, hop_len=hl)
        ml = min(len(env_pred), len(env_gt))
        if ml > 2 and np.std(env_pred[:ml]) > 1e-7 and np.std(env_gt[:ml]) > 1e-7:
            r_env, _ = pearsonr(env_pred[:ml], env_gt[:ml])
            env_corrs[f'env_corr_{fl}'] = float(r_env) if not np.isnan(r_env) else 0.0
        else:
            env_corrs[f'env_corr_{fl}'] = 0.0

    return {
        'pearson_correlation': pcc,
        'rmse': rmse,
        'snr_db': snr_db,
        'peak_amplitude_ratio': peak_ratio,
        'rms_ratio': rms_ratio,
        'peak_gt': peak_gt,
        'peak_pred': peak_pred,
        'rms_gt': rms_gt,
        'rms_pred': rms_pred,
        **env_corrs
    }

def compute_frequency_metrics(pred: np.ndarray, gt: np.ndarray, sr: int = 16000) -> dict:
    """Computes frequency-domain metrics using Welch's Power Spectral Density."""
    pred = _to_numpy(pred)
    gt = _to_numpy(gt)
    
    min_len = min(len(pred), len(gt))
    pred = pred[:min_len]
    gt = gt[:min_len]
    
    nperseg = min(1024, len(pred))
    freqs, psd_gt = signal.welch(gt, fs=sr, nperseg=nperseg, noverlap=nperseg//2)
    _, psd_pred = signal.welch(pred, fs=sr, nperseg=nperseg, noverlap=nperseg//2)
    
    psd_gt = np.maximum(psd_gt, 1e-12)
    psd_pred = np.maximum(psd_pred, 1e-12)
    
    # 1. Log Spectral Distance (LSD) across spectrum (in dB)
    log_gt = 10.0 * np.log10(psd_gt)
    log_pred = 10.0 * np.log10(psd_pred)
    lsd = float(np.mean(np.abs(log_gt - log_pred)))
    
    # 2. Tactile band (50 - 300 Hz) LSD
    tactile_mask = (freqs >= 50.0) & (freqs <= 300.0)
    if np.any(tactile_mask):
        lsd_tactile = float(np.mean(np.abs(log_gt[tactile_mask] - log_pred[tactile_mask])))
        trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        if trapz_func is None:
            from scipy.integrate import trapezoid as trapz_func
        energy_tactile_gt = float(trapz_func(psd_gt[tactile_mask], freqs[tactile_mask]))
        energy_tactile_pred = float(trapz_func(psd_pred[tactile_mask], freqs[tactile_mask]))
        tactile_band_energy_ratio = float(energy_tactile_pred / (energy_tactile_gt + 1e-10))
    else:
        lsd_tactile = lsd
        tactile_band_energy_ratio = 1.0
        
    # 3. Spectral Convergence (SC)
    sc = float(np.linalg.norm(psd_gt - psd_pred) / (np.linalg.norm(psd_gt) + 1e-8))
    
    # 4. Spectral Centroid (Hz)
    centroid_gt = float(np.sum(freqs * psd_gt) / (np.sum(psd_gt) + 1e-8))
    centroid_pred = float(np.sum(freqs * psd_pred) / (np.sum(psd_pred) + 1e-8))
    centroid_delta_hz = float(abs(centroid_pred - centroid_gt))
    
    # 5. Spectral Flatness (Wiener entropy: geometric mean / arithmetic mean)
    def spectral_flatness(p):
        log_p = np.log(p + 1e-12)
        geom_mean = np.exp(np.mean(log_p))
        arith_mean = np.mean(p) + 1e-12
        return float(geom_mean / arith_mean)
        
    flatness_gt = spectral_flatness(psd_gt)
    flatness_pred = spectral_flatness(psd_pred)
    flatness_delta = float(abs(flatness_pred - flatness_gt))
    
    # 6. PSD Cosine Similarity
    norm_gt = np.linalg.norm(psd_gt) + 1e-8
    norm_pred = np.linalg.norm(psd_pred) + 1e-8
    psd_cosine_sim = float(np.dot(psd_gt, psd_pred) / (norm_gt * norm_pred))
    
    return {
        'log_spectral_distance': lsd,
        'lsd_tactile_band': lsd_tactile,
        'tactile_band_energy_ratio': tactile_band_energy_ratio,
        'spectral_convergence': sc,
        'spectral_centroid_gt_hz': centroid_gt,
        'spectral_centroid_pred_hz': centroid_pred,
        'spectral_centroid_delta_hz': centroid_delta_hz,
        'spectral_flatness_gt': flatness_gt,
        'spectral_flatness_pred': flatness_pred,
        'spectral_flatness_delta': flatness_delta,
        'psd_cosine_similarity': psd_cosine_sim,
    }

def compute_perceptual_metrics(pred: np.ndarray, gt: np.ndarray, sr: int = 16000) -> dict:
    """Computes psychophysically weighted haptic quality metrics."""
    pred = _to_numpy(pred)
    gt = _to_numpy(gt)
    
    min_len = min(len(pred), len(gt))
    pred = pred[:min_len]
    gt = gt[:min_len]
    
    nperseg = min(1024, len(pred))
    freqs, psd_gt = signal.welch(gt, fs=sr, nperseg=nperseg, noverlap=nperseg//2)
    _, psd_pred = signal.welch(pred, fs=sr, nperseg=nperseg, noverlap=nperseg//2)
    
    weights = tactile_sensitivity_weighting(freqs)
    
    # Frequency-Weighted MSE in spectral domain
    weighted_diff = weights * (psd_pred - psd_gt)
    freq_weighted_mse = float(np.mean(weighted_diff ** 2))
    
    # SPQI-Lite: Psychophysically weighted log spectral distance
    log_gt = 10.0 * np.log10(np.maximum(psd_gt, 1e-12))
    log_pred = 10.0 * np.log10(np.maximum(psd_pred, 1e-12))
    spqi_lite = float(np.sum(weights * np.abs(log_gt - log_pred)) / (np.sum(weights) + 1e-8))
    
    return {
        'frequency_weighted_mse': freq_weighted_mse,
        'spqi_lite': spqi_lite,
    }

def compute_all_metrics(pred, gt, sr: int = 16000) -> dict:
    """
    Computes all time-domain, frequency-domain, and perceptual metrics.
    Accepts 1D numpy arrays or PyTorch tensors.
    """
    pred_np = _to_numpy(pred)
    gt_np = _to_numpy(gt)
    
    time_m = compute_time_metrics(pred_np, gt_np)
    freq_m = compute_frequency_metrics(pred_np, gt_np, sr=sr)
    perc_m = compute_perceptual_metrics(pred_np, gt_np, sr=sr)
    
    return {**time_m, **freq_m, **perc_m}

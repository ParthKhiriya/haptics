import json
from pathlib import Path
import numpy as np
import torch
from scipy import signal
from scipy.linalg import sqrtm
from scipy.signal import hilbert
from scipy.stats import pearsonr


def _to_numpy(x) -> np.ndarray:
    """Ensure input is a 1D float64 numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().squeeze().numpy()
    elif isinstance(x, np.ndarray):
        x = x.squeeze()
    return np.asarray(x, dtype=np.float64).flatten()


def compute_rms_envelope(x: np.ndarray, frame_len: int = 256, hop_len: int = 128) -> np.ndarray:
    """Compute moving RMS envelope."""
    x = _to_numpy(x)
    if len(x) < frame_len:
        return np.array([np.sqrt(np.mean(x**2) + 1e-10)], dtype=np.float32)
    pad_len = frame_len // 2
    x_padded = np.pad(x, (pad_len, pad_len), mode="reflect")
    window = np.ones(frame_len, dtype=np.float32) / frame_len
    env = np.sqrt(np.maximum(0, np.convolve(x_padded**2, window, mode="valid")) + 1e-10)
    return env[::hop_len].astype(np.float32)


def tactile_sensitivity_weighting(freqs: np.ndarray) -> np.ndarray:
    """
    Psychophysical frequency weighting W(f) for vibrotactile perception.
    Peaks around 200-250 Hz (Pacinian channel maximum sensitivity)
    and drops towards low (<20 Hz) and high (>500 Hz) frequencies.
    """
    freqs = np.asarray(freqs, dtype=np.float32)
    w = np.zeros_like(freqs)
    valid = freqs > 5.0
    center_f = 220.0
    sigma_log = np.log(2.3)
    w[valid] = np.exp(-((np.log(freqs[valid]) - np.log(center_f)) ** 2) / (2.0 * sigma_log ** 2))
    w = w / (np.max(w) + 1e-8)
    return w


def find_optimal_micro_shift(pred: np.ndarray, target: np.ndarray, max_shift: int) -> np.ndarray:
    """
    Aligns `pred` to `target` within [-max_shift, +max_shift] samples by maximizing zero-mean correlation.
    """
    if max_shift <= 0:
        return pred
    best_corr = -1e9
    best_shifted = pred
    t_zm = target - np.mean(target)
    norm_t = np.linalg.norm(t_zm) + 1e-8
    for shift in range(-max_shift, max_shift + 1):
        if shift < 0:
            p_s = np.pad(pred[-shift:], (0, -shift), mode="edge")
        elif shift > 0:
            p_s = np.pad(pred[:-shift], (shift, 0), mode="edge")
        else:
            p_s = pred
        p_zm = p_s - np.mean(p_s)
        norm_p = np.linalg.norm(p_zm) + 1e-8
        corr = float(np.sum(p_zm * t_zm) / (norm_p * norm_t))
        if corr > best_corr:
            best_corr = corr
            best_shifted = p_s
    return best_shifted


def compute_single_pair_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    fs: int = 16000,
    max_align_ms: float = 1.5,
) -> dict[str, float]:
    """
    Computes 20+ publication-level vibrotactile evaluation metrics for a single tap pair.
    Covers Time-domain (raw and micro-aligned), Transient & Impact Physics,
    Spectral & Modal Ringing, and Psychophysical Mechanoreceptor Bands.
    """
    pred = _to_numpy(pred)
    target = _to_numpy(target)
    T = min(len(pred), len(target))
    pred, target = pred[:T], target[:T]
    eps = 1e-8

    max_shift = int(round(max_align_ms * 0.001 * fs))
    pred_aligned = find_optimal_micro_shift(pred, target, max_shift=max_shift)

    def _time_metrics(p: np.ndarray, t: np.ndarray) -> tuple[float, float, float, float, float]:
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        mae = float(np.mean(np.abs(p - t)))
        p_zm = p - np.mean(p)
        t_zm = t - np.mean(t)
        norm_p = np.linalg.norm(p_zm) + eps
        norm_t = np.linalg.norm(t_zm) + eps
        r = float(np.sum(p_zm * t_zm) / (norm_p * norm_t))
        sig_pwr = np.sum(t ** 2) + eps
        nse_pwr = np.sum((p - t) ** 2) + eps
        snr = float(10.0 * np.log10(sig_pwr / nse_pwr))

        denom = np.sum(t_zm ** 2) + eps
        alpha = np.sum(p_zm * t_zm) / denom
        s_target = alpha * t_zm
        e_noise = p_zm - s_target
        si_sdr = float(10.0 * np.log10((np.sum(s_target ** 2) + eps) / (np.sum(e_noise ** 2) + eps)))
        return rmse, mae, r, snr, si_sdr

    rmse_raw, mae_raw, r_raw, snr_raw, sisdr_raw = _time_metrics(pred, target)
    rmse_al, mae_al, r_al, snr_al, sisdr_al = _time_metrics(pred_aligned, target)

    # 2. Transient & Impact Physics Metrics
    env_p = np.abs(hilbert(pred_aligned - np.mean(pred_aligned)))
    env_t = np.abs(hilbert(target - np.mean(target)))
    env_rmse = float(np.sqrt(np.mean((env_p - env_t) ** 2)))
    if np.std(env_p) > eps and np.std(env_t) > eps:
        env_r = float(np.corrcoef(env_p, env_t)[0, 1])
        if np.isnan(env_r):
            env_r = 0.0
    else:
        env_r = 0.0

    peak_p = float(np.max(np.abs(pred_aligned)) + eps)
    peak_t = float(np.max(np.abs(target)) + eps)
    peak_rel_err = float(abs(peak_p - peak_t) / peak_t)

    rms_p = float(np.sqrt(np.mean(pred_aligned ** 2) + eps))
    rms_t = float(np.sqrt(np.mean(target ** 2) + eps))
    cf_p_db = float(20.0 * np.log10(peak_p / rms_p))
    cf_t_db = float(20.0 * np.log10(peak_t / rms_t))
    crest_factor_err_db = float(abs(cf_p_db - cf_t_db))

    # Attack slope error (max positive derivative in impact region)
    slope_p = float(np.max(np.diff(pred_aligned)) * fs)
    slope_t = float(np.max(np.diff(target)) * fs + eps)
    attack_slope_rel_err = float(abs(slope_p - slope_t) / (abs(slope_t) + eps))

    # Schroeder Energy Decay Curve (EDC) RMSE in dB
    edc_p = np.cumsum((pred_aligned ** 2)[::-1])[::-1]
    edc_t = np.cumsum((target ** 2)[::-1])[::-1]
    edc_p_db = 10.0 * np.log10(np.clip(edc_p / (edc_p[0] + eps), 1e-4, 1.0))
    edc_t_db = 10.0 * np.log10(np.clip(edc_t / (edc_t[0] + eps), 1e-4, 1.0))
    edc_rmse_db = float(np.sqrt(np.mean((edc_p_db - edc_t_db) ** 2)))

    # 3. Spectral & Modal Ringing Metrics
    P_f = np.abs(np.fft.rfft(pred_aligned)) + eps
    T_f = np.abs(np.fft.rfft(target)) + eps
    freqs = np.fft.rfftfreq(T, d=1.0 / float(fs))

    lsd = float(np.sqrt(np.mean((10.0 * np.log10((P_f ** 2) / (T_f ** 2))) ** 2)))

    # Spectral Centroid Error (Hz)
    cent_p = float(np.sum(freqs * P_f) / np.sum(P_f))
    cent_t = float(np.sum(freqs * T_f) / np.sum(T_f))
    centroid_err_hz = float(abs(cent_p - cent_t))

    # PSD Jensen-Shannon Divergence
    psd_p = (P_f ** 2) / np.sum(P_f ** 2)
    psd_t = (T_f ** 2) / np.sum(T_f ** 2)
    m = 0.5 * (psd_p + psd_t)
    psd_jsd = float(
        0.5 * np.sum(psd_p * np.log2((psd_p + eps) / (m + eps)))
        + 0.5 * np.sum(psd_t * np.log2((psd_t + eps) / (m + eps)))
    )

    # 4. Psychophysical Mechanoreceptor Band Errors (dB LSD in RA and PC bands)
    ra_mask = (freqs >= 5.0) & (freqs <= 100.0)
    pc_mask = (freqs >= 40.0) & (freqs <= 800.0)
    meissner_ra_lsd = float(
        np.sqrt(np.mean((20.0 * np.log10(P_f[ra_mask] / T_f[ra_mask])) ** 2))
        if np.any(ra_mask) else lsd
    )
    pacinian_pc_lsd = float(
        np.sqrt(np.mean((20.0 * np.log10(P_f[pc_mask] / T_f[pc_mask])) ** 2))
        if np.any(pc_mask) else lsd
    )

    return {
        "waveform_rmse_raw": rmse_raw,
        "waveform_rmse_aligned": rmse_al,
        "waveform_mae_aligned": mae_al,
        "pearson_r_raw": r_raw,
        "pearson_r_aligned": r_al,
        "snr_db_raw": snr_raw,
        "snr_db_aligned": snr_al,
        "si_sdr_db_raw": sisdr_raw,
        "si_sdr_db_aligned": sisdr_al,
        "envelope_rmse": env_rmse,
        "envelope_pearson_r": env_r,
        "peak_amplitude_rel_err": peak_rel_err,
        "crest_factor_err_db": crest_factor_err_db,
        "attack_slope_rel_err": attack_slope_rel_err,
        "schroeder_edc_rmse_db": edc_rmse_db,
        "log_spectral_distance_db": lsd,
        "spectral_centroid_err_hz": centroid_err_hz,
        "psd_jsd_bits": psd_jsd,
        "meissner_ra_band_lsd_db": meissner_ra_lsd,
        "pacinian_pc_band_lsd_db": pacinian_pc_lsd,
    }


def compute_frechet_haptic_distance(
    preds: list[np.ndarray],
    targets: list[np.ndarray],
    fs: int = 16000,
) -> float:
    """
    Computes Fréchet Haptic Distance (FHD) in physics-informed spectral-decay feature space.
    """
    def _embed_batch(signals: list[np.ndarray]) -> np.ndarray:
        feats = []
        for s in signals:
            s = _to_numpy(s)
            P = np.log10(np.abs(np.fft.rfft(s)) + 1e-6)
            # Interpolate spectrum to 48 bins
            x_old = np.linspace(0.0, 1.0, len(P))
            x_new = np.linspace(0.0, 1.0, 48)
            p_bins = np.interp(x_new, x_old, P)
            # Add Schroeder decay quantiles (16 bins)
            edc = np.cumsum((s ** 2)[::-1])[::-1]
            edc_db = 10.0 * np.log10(np.clip(edc / (edc[0] + 1e-8), 1e-4, 1.0))
            e_bins = np.interp(np.linspace(0.0, 1.0, 16), np.linspace(0.0, 1.0, len(edc_db)), edc_db)
            feats.append(np.concatenate([p_bins, e_bins]))
        return np.vstack(feats)

    if len(preds) < 4:
        return 0.0
    emb_p = _embed_batch(preds)
    emb_t = _embed_batch(targets)

    mu_p, mu_t = np.mean(emb_p, axis=0), np.mean(emb_t, axis=0)
    cov_p = np.cov(emb_p, rowvar=False) + np.eye(emb_p.shape[1]) * 1e-5
    cov_t = np.cov(emb_t, rowvar=False) + np.eye(emb_t.shape[1]) * 1e-5

    covmean = sqrtm(cov_p @ cov_t)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fhd = float(np.sum((mu_p - mu_t) ** 2) + np.trace(cov_p + cov_t - 2.0 * covmean))
    return max(0.0, fhd)


def export_publication_eval_report(
    preds: list[np.ndarray],
    targets: list[np.ndarray],
    fs: int = 16000,
    output_dir: str = "results",
) -> dict[str, float]:
    """
    Runs full publication evaluation across predictions and exports JSON and Markdown tables.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    per_sample = [compute_single_pair_metrics(p, t, fs=fs) for p, t in zip(preds, targets)]
    keys = list(per_sample[0].keys())
    summary = {}
    for k in keys:
        vals = [m[k] for m in per_sample]
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals))

    summary["frechet_haptic_distance"] = compute_frechet_haptic_distance(preds, targets, fs=fs)

    # Save JSON report
    json_file = out_path / "eval_metrics_report.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save Markdown summary table
    md_lines = [
        "# Controlled-Tap Audio-to-Haptics Publication Evaluation Report",
        "",
        "| Metric | Mean | Std | Direction |",
        "| :--- | :---: | :---: | :---: |",
    ]
    higher_better = {
        "pearson_r_raw",
        "pearson_r_aligned",
        "snr_db_raw",
        "snr_db_aligned",
        "si_sdr_db_raw",
        "si_sdr_db_aligned",
        "envelope_pearson_r",
    }
    for k in keys:
        direction = "↑ Higher is better" if k in higher_better else "↓ Lower is better"
        md_lines.append(f"| `{k}` | {summary[f'{k}_mean']:.4f} | ±{summary[f'{k}_std']:.4f} | {direction} |")
    md_lines.append(f"| `frechet_haptic_distance` | {summary['frechet_haptic_distance']:.4f} | - | ↓ Lower is better |")

    md_file = out_path / "eval_summary.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return summary


def compute_time_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Computes time-domain metrics comparing prediction to ground truth."""
    pred = _to_numpy(pred)
    gt = _to_numpy(gt)
    min_len = min(len(pred), len(gt))
    pred = pred[:min_len]
    gt = gt[:min_len]

    if np.std(pred) < 1e-7 or np.std(gt) < 1e-7:
        pcc = 0.0
    else:
        r, _ = pearsonr(pred, gt)
        pcc = float(r) if not np.isnan(r) else 0.0

    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    noise_power = np.sum((gt - pred) ** 2)
    signal_power = np.sum(gt ** 2)
    snr_db = float(10.0 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10)))

    peak_pred = float(np.max(np.abs(pred)))
    peak_gt = float(np.max(np.abs(gt)))
    peak_ratio = float(peak_pred / (peak_gt + 1e-8))

    rms_pred = float(np.sqrt(np.mean(pred ** 2)))
    rms_gt = float(np.sqrt(np.mean(gt ** 2)))
    rms_ratio = float(rms_pred / (rms_gt + 1e-8))

    env_corrs = {}
    for fl in [64, 256, 1024]:
        hl = max(1, fl // 2)
        env_pred = compute_rms_envelope(pred, frame_len=fl, hop_len=hl)
        env_gt = compute_rms_envelope(gt, frame_len=fl, hop_len=hl)
        ml = min(len(env_pred), len(env_gt))
        if ml > 2 and np.std(env_pred[:ml]) > 1e-7 and np.std(env_gt[:ml]) > 1e-7:
            r_env, _ = pearsonr(env_pred[:ml], env_gt[:ml])
            env_corrs[f"env_corr_{fl}"] = float(r_env) if not np.isnan(r_env) else 0.0
        else:
            env_corrs[f"env_corr_{fl}"] = 0.0

    return {
        "pearson_correlation": pcc,
        "rmse": rmse,
        "snr_db": snr_db,
        "peak_amplitude_ratio": peak_ratio,
        "rms_ratio": rms_ratio,
        "peak_gt": peak_gt,
        "peak_pred": peak_pred,
        "rms_gt": rms_gt,
        "rms_pred": rms_pred,
        **env_corrs,
    }


def compute_frequency_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    sr: int = 16000,
    tactile_low: float = 50.0,
    tactile_high: float = 300.0,
) -> dict:
    """Computes frequency-domain metrics using Welch's Power Spectral Density."""
    pred = _to_numpy(pred)
    gt = _to_numpy(gt)
    min_len = min(len(pred), len(gt))
    pred = pred[:min_len]
    gt = gt[:min_len]

    nperseg = min(1024, len(pred))
    freqs, psd_gt = signal.welch(gt, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    _, psd_pred = signal.welch(pred, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)

    psd_gt = np.maximum(psd_gt, 1e-12)
    psd_pred = np.maximum(psd_pred, 1e-12)

    log_gt = 10.0 * np.log10(psd_gt)
    log_pred = 10.0 * np.log10(psd_pred)
    lsd = float(np.mean(np.abs(log_gt - log_pred)))

    tactile_mask = (freqs >= tactile_low) & (freqs <= tactile_high)
    if np.any(tactile_mask):
        lsd_tactile = float(np.mean(np.abs(log_gt[tactile_mask] - log_pred[tactile_mask])))
        trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
        if trapz_func is None:
            from scipy.integrate import trapezoid as trapz_func
        energy_tactile_gt = float(trapz_func(psd_gt[tactile_mask], freqs[tactile_mask]))
        energy_tactile_pred = float(trapz_func(psd_pred[tactile_mask], freqs[tactile_mask]))
        tactile_band_energy_ratio = float(energy_tactile_pred / (energy_tactile_gt + 1e-10))
    else:
        lsd_tactile = lsd
        tactile_band_energy_ratio = 1.0

    sc = float(np.linalg.norm(psd_gt - psd_pred) / (np.linalg.norm(psd_gt) + 1e-8))
    centroid_gt = float(np.sum(freqs * psd_gt) / (np.sum(psd_gt) + 1e-8))
    centroid_pred = float(np.sum(freqs * psd_pred) / (np.sum(psd_pred) + 1e-8))
    centroid_delta_hz = float(abs(centroid_pred - centroid_gt))

    def spectral_flatness(p):
        log_p = np.log(p + 1e-12)
        geom_mean = np.exp(np.mean(log_p))
        arith_mean = np.mean(p) + 1e-12
        return float(geom_mean / arith_mean)

    flatness_gt = spectral_flatness(psd_gt)
    flatness_pred = spectral_flatness(psd_pred)
    flatness_delta = float(abs(flatness_pred - flatness_gt))

    norm_gt = np.linalg.norm(psd_gt) + 1e-8
    norm_pred = np.linalg.norm(psd_pred) + 1e-8
    psd_cosine_sim = float(np.dot(psd_gt, psd_pred) / (norm_gt * norm_pred))

    return {
        "log_spectral_distance": lsd,
        "lsd_tactile_band": lsd_tactile,
        "tactile_band_energy_ratio": tactile_band_energy_ratio,
        "spectral_convergence": sc,
        "spectral_centroid_gt_hz": centroid_gt,
        "spectral_centroid_pred_hz": centroid_pred,
        "spectral_centroid_delta_hz": centroid_delta_hz,
        "spectral_flatness_gt": flatness_gt,
        "spectral_flatness_pred": flatness_pred,
        "spectral_flatness_delta": flatness_delta,
        "psd_cosine_similarity": psd_cosine_sim,
    }


def compute_perceptual_metrics(pred: np.ndarray, gt: np.ndarray, sr: int = 16000) -> dict:
    """Computes psychophysically weighted haptic quality metrics."""
    pred = _to_numpy(pred)
    gt = _to_numpy(gt)
    min_len = min(len(pred), len(gt))
    pred = pred[:min_len]
    gt = gt[:min_len]

    nperseg = min(1024, len(pred))
    freqs, psd_gt = signal.welch(gt, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    _, psd_pred = signal.welch(pred, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)

    weights = tactile_sensitivity_weighting(freqs)
    weighted_diff = weights * (psd_pred - psd_gt)
    freq_weighted_mse = float(np.mean(weighted_diff ** 2))

    log_gt = 10.0 * np.log10(np.maximum(psd_gt, 1e-12))
    log_pred = 10.0 * np.log10(np.maximum(psd_pred, 1e-12))
    spqi_lite = float(np.sum(weights * np.abs(log_gt - log_pred)) / (np.sum(weights) + 1e-8))

    return {
        "frequency_weighted_mse": freq_weighted_mse,
        "spqi_lite": spqi_lite,
    }


def compute_all_metrics(
    pred,
    gt,
    sr: int = 16000,
    tactile_low: float = 50.0,
    tactile_high: float = 300.0,
) -> dict:
    """
    Computes all time-domain, frequency-domain, and perceptual metrics.
    Merges legacy metrics and publication single-pair metrics for complete coverage.
    """
    pred_np = _to_numpy(pred)
    gt_np = _to_numpy(gt)

    time_m = compute_time_metrics(pred_np, gt_np)
    freq_m = compute_frequency_metrics(
        pred_np, gt_np, sr=sr, tactile_low=tactile_low, tactile_high=tactile_high
    )
    perc_m = compute_perceptual_metrics(pred_np, gt_np, sr=sr)
    pub_m = compute_single_pair_metrics(pred_np, gt_np, fs=sr)

    return {**time_m, **freq_m, **perc_m, **pub_m}

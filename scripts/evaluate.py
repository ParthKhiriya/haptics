import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm

# Inject project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.encodec_wrapper import FrozenAudioEncoder
from src.models.decoder import TactileSEANetDecoder
from src.evaluation.metrics import (
    compute_all_metrics, 
    compute_rms_envelope, 
    tactile_sensitivity_weighting
)

CLASS_MAP = {
    'C1': 'Wood',
    'C2': 'Paper/Fabric',
    'C3': 'Plastic/Rubber',
    'C4': 'Foam',
    'C5': 'Carpet/Leather',
    'C6': 'Stone/Ceramic',
    'C7': 'Metal',
    'C8': 'Glass'
}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Sound2Hap Cross-Modal Haptic Synthesis")
    parser.add_argument("--checkpoint", type=str, default=str(PROJECT_ROOT / "checkpoints" / "best_sean_decoder.pt"),
                        help="Path to trained decoder checkpoint")
    parser.add_argument("--processed_dir", type=str, default=str(PROJECT_ROOT / "data" / "processed"),
                        help="Directory with processed .npy pairs")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "evaluation_results"),
                        help="Directory to save evaluation reports and plots")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to evaluate (default: evaluate all)")
    parser.add_argument("--num_plot_samples", type=int, default=6,
                        help="Number of individual samples to plot in figure")
    return parser.parse_args()

def evaluate():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    working_sr = checkpoint.get('working_sr', 16000)
    window_size = checkpoint.get('window_size', 16000)
    print(f"Loaded checkpoint from {ckpt_path.name} (Epoch {checkpoint.get('epoch', '?')}, Val Loss: {checkpoint.get('val_loss', 0.0):.4f})")

    # Discover validation files (reproduce 80/20 split based on seed 42)
    processed_dir = Path(args.processed_dir)
    audio_files = sorted(list(processed_dir.rglob("*_audio.npy")))
    pairs = []
    for af in audio_files:
        tf = af.parent / af.name.replace("_audio.npy", "_tactile.npy")
        if tf.exists():
            pairs.append((af, tf))
            
    if not pairs:
        print(f"Error: No paired files found in {processed_dir}")
        return
        
    np.random.seed(42)
    indices = np.random.permutation(len(pairs))
    train_size = int(0.8 * len(pairs))
    val_indices = indices[train_size:]
    val_pairs = [pairs[i] for i in val_indices]
    
    if args.max_samples:
        val_pairs = val_pairs[:args.max_samples]
        
    print(f"Evaluating {len(val_pairs)} validation samples on device: {device}...")

    # Load Models
    encoder = FrozenAudioEncoder(target_bandwidth=1.5, device=device, input_sr=working_sr)
    decoder = TactileSEANetDecoder(input_dim=128, output_dim=1, target_length=window_size).to(device)
    
    state_dict = checkpoint['model_state_dict']
    # If checkpoint had old output_scale parameter, ignore it smoothly
    if 'output_scale' in state_dict:
        del state_dict['output_scale']
        
    decoder.load_state_dict(state_dict, strict=False)
    decoder.eval()

    all_results = []
    class_results = {c: [] for c in CLASS_MAP.keys()}
    
    sample_plots_data = []
    all_psd_gt = []
    all_psd_pred = []
    psd_freqs = None

    for i, (aud_path, tac_path) in enumerate(tqdm(val_pairs, desc="Evaluating")):
        # Extract class code from filename e.g. C1_S1...
        stem = aud_path.stem.replace("_audio", "")
        class_code = stem.split("_")[0] if "_" in stem else "Other"
        
        audio = np.load(aud_path).flatten()
        tactile = np.load(tac_path).flatten()
        
        min_len = min(len(audio), len(tactile))
        if min_len > window_size:
            audio = audio[:window_size]
            tactile = tactile[:window_size]
        else:
            audio = np.pad(audio[:min_len], (0, max(0, window_size - min_len)))
            tactile = np.pad(tactile[:min_len], (0, max(0, window_size - min_len)))
            
        # Audio normalized for EnCodec
        audio_norm = audio - audio.mean()
        audio_norm = audio_norm / (np.abs(audio_norm).max() + 1e-8)
        
        # Tactile is already processed & z-scored; do not re-normalize
        audio_t = torch.from_numpy(audio_norm).float().unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            latent = encoder(audio_t)
            pred_t = decoder(latent)
            
        pred = pred_t.squeeze().cpu().numpy()
        gt = tactile
        
        metrics = compute_all_metrics(pred, gt, sr=working_sr)
        metrics['filename'] = aud_path.name
        metrics['class'] = class_code
        metrics['class_name'] = CLASS_MAP.get(class_code, 'Unknown')
        
        all_results.append(metrics)
        if class_code in class_results:
            class_results[class_code].append(metrics)
            
        # Compute PSD for aggregation
        nperseg = min(1024, len(pred))
        freqs, psd_gt = signal.welch(gt, fs=working_sr, nperseg=nperseg, noverlap=nperseg//2)
        _, psd_p = signal.welch(pred, fs=working_sr, nperseg=nperseg, noverlap=nperseg//2)
        if psd_freqs is None:
            psd_freqs = freqs
        all_psd_gt.append(psd_gt)
        all_psd_pred.append(psd_p)
        
        if len(sample_plots_data) < args.num_plot_samples:
            sample_plots_data.append({
                'name': stem,
                'class': CLASS_MAP.get(class_code, class_code),
                'pred': pred,
                'gt': gt,
                'psd_gt': psd_gt,
                'psd_pred': psd_p,
                'freqs': freqs,
                'metrics': metrics
            })

    # ─── 1. Aggregate Metrics ───
    metric_keys = [
        'pearson_correlation', 'rmse', 'snr_db', 'peak_amplitude_ratio', 'rms_ratio',
        'env_corr_64', 'env_corr_256', 'env_corr_1024',
        'log_spectral_distance', 'lsd_tactile_band', 'tactile_band_energy_ratio',
        'spectral_convergence', 'spectral_centroid_delta_hz', 'spectral_flatness_delta',
        'psd_cosine_similarity', 'frequency_weighted_mse', 'spqi_lite'
    ]
    
    summary = {}
    for k in metric_keys:
        vals = [r[k] for r in all_results if k in r and not np.isnan(r[k])]
        summary[k] = {
            'mean': float(np.mean(vals)) if vals else 0.0,
            'std': float(np.std(vals)) if vals else 0.0,
            'median': float(np.median(vals)) if vals else 0.0,
        }
        
    per_class_summary = {}
    for c_code, c_name in CLASS_MAP.items():
        c_items = class_results.get(c_code, [])
        if not c_items:
            continue
        per_class_summary[f"{c_code} ({c_name})"] = {
            'count': len(c_items),
            'pcc': float(np.mean([x['pearson_correlation'] for x in c_items])),
            'rmse': float(np.mean([x['rmse'] for x in c_items])),
            'snr_db': float(np.mean([x['snr_db'] for x in c_items])),
            'peak_ratio': float(np.mean([x['peak_amplitude_ratio'] for x in c_items])),
            'lsd_tactile': float(np.mean([x['lsd_tactile_band'] for x in c_items])),
            'psd_cosine_sim': float(np.mean([x['psd_cosine_similarity'] for x in c_items])),
            'spqi_lite': float(np.mean([x['spqi_lite'] for x in c_items]))
        }

    # ─── 2. Print Summary Table ───
    print("\n" + "="*80)
    print("                    SOUND2HAP EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Category':<16} | {'Metric':<28} | {'Mean ± Std':<20} | {'Ideal':<8}")
    print("-" * 80)
    
    table_items = [
        ("Time Domain", "Pearson Correlation (PCC)", "pearson_correlation", "-> 1.0"),
        ("Time Domain", "RMSE", "rmse", "-> 0.0"),
        ("Time Domain", "SNR (dB)", "snr_db", "Higher"),
        ("Time Domain", "Peak Amplitude Ratio", "peak_amplitude_ratio", "-> 1.0"),
        ("Time Domain", "RMS Energy Ratio", "rms_ratio", "-> 1.0"),
        ("Time Domain", "Envelope Corr (64)", "env_corr_64", "-> 1.0"),
        ("Time Domain", "Envelope Corr (256)", "env_corr_256", "-> 1.0"),
        ("Time Domain", "Envelope Corr (1024)", "env_corr_1024", "-> 1.0"),
        ("Spectral", "Log Spectral Dist (LSD dB)", "log_spectral_distance", "-> 0.0"),
        ("Spectral", "Tactile Band (50-300Hz) LSD", "lsd_tactile_band", "-> 0.0"),
        ("Spectral", "Tactile Band Energy Ratio", "tactile_band_energy_ratio", "-> 1.0"),
        ("Spectral", "Spectral Convergence", "spectral_convergence", "-> 0.0"),
        ("Spectral", "Spectral Centroid Delta (Hz)", "spectral_centroid_delta_hz", "-> 0.0"),
        ("Spectral", "Spectral Flatness Delta", "spectral_flatness_delta", "-> 0.0"),
        ("Spectral", "PSD Cosine Similarity", "psd_cosine_similarity", "-> 1.0"),
        ("Perceptual", "Frequency-Weighted MSE", "frequency_weighted_mse", "-> 0.0"),
        ("Perceptual", "SPQI-Lite (Weighted LSD)", "spqi_lite", "-> 0.0"),
    ]
    
    for cat, name, key, ideal in table_items:
        m_val = summary[key]['mean']
        s_val = summary[key]['std']
        print(f"{cat:<16} | {name:<28} | {m_val:7.4f} ± {s_val:6.4f}     | {ideal:<8}")
        
    print("\n" + "="*80)
    print("                    PER-CLASS PERFORMANCE BREAKDOWN")
    print("="*80)
    print(f"{'Class':<22} | {'Count':<5} | {'PCC':<6} | {'RMSE':<6} | {'SNR(dB)':<7} | {'PeakRatio':<9} | {'TactileLSD':<10} | {'CosSim':<6}")
    print("-" * 80)
    for c_label, d in per_class_summary.items():
        print(f"{c_label:<22} | {d['count']:<5} | {d['pcc']:<6.3f} | {d['rmse']:<6.3f} | {d['snr_db']:<7.2f} | {d['peak_ratio']:<9.3f} | {d['lsd_tactile']:<10.2f} | {d['psd_cosine_sim']:<6.3f}")
    print("="*80 + "\n")

    # ─── 3. Save JSON Report ───
    report = {
        'checkpoint': ckpt_path.name,
        'num_samples_evaluated': len(all_results),
        'working_sr': working_sr,
        'summary': summary,
        'per_class_summary': per_class_summary,
        'per_sample_results': all_results
    }
    with open(out_dir / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[*] Full report saved to: {out_dir / 'evaluation_report.json'}")

    # ─── 4. Plot 1: Spectral PSD Comparison ───
    n_plots = len(sample_plots_data)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3.2 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    fig.suptitle("Sound2Hap: Power Spectral Density Comparison (Welch PSD)", fontsize=16)

    for idx, (ax, s) in enumerate(zip(axes, sample_plots_data)):
        freqs = s['freqs']
        psd_gt_db = 10.0 * np.log10(np.maximum(s['psd_gt'], 1e-12))
        psd_pred_db = 10.0 * np.log10(np.maximum(s['psd_pred'], 1e-12))
        
        # Highlight 50-300 Hz tactile band
        ax.axvspan(50, 300, color='green', alpha=0.15, label='Tactile Band (50-300 Hz)' if idx == 0 else "")
        ax.plot(freqs, psd_gt_db, color='blue', label='Ground Truth' if idx == 0 else "", alpha=0.85, linewidth=1.5)
        ax.plot(freqs, psd_pred_db, color='orange', label='Predicted' if idx == 0 else "", alpha=0.85, linewidth=1.5)
        
        # Annotate centroids
        c_gt = s['metrics']['spectral_centroid_gt_hz']
        c_pr = s['metrics']['spectral_centroid_pred_hz']
        ax.axvline(c_gt, color='blue', linestyle='--', alpha=0.6, label='GT Centroid' if idx == 0 else "")
        ax.axvline(c_pr, color='orange', linestyle='--', alpha=0.6, label='Pred Centroid' if idx == 0 else "")
        
        pcc = s['metrics']['pearson_correlation']
        pr = s['metrics']['peak_amplitude_ratio']
        lsd = s['metrics']['lsd_tactile_band']
        ax.set_title(f"Sample: {s['name']} [{s['class']}] | PCC: {pcc:.3f} | Peak Ratio: {pr:.2f}x | Tactile LSD: {lsd:.2f} dB", fontsize=11)
        ax.set_ylabel("Power/Freq (dB/Hz)")
        ax.set_xlim(20, 2000)
        ax.set_xscale('log')
        ax.grid(True, which='both', linestyle=':', alpha=0.6)
        
    axes[0].legend(loc='upper right')
    axes[-1].set_xlabel("Frequency (Hz) [Log Scale]")
    plt.tight_layout()
    plt.savefig(out_dir / "evaluation_spectral.png", dpi=300)
    plt.close()
    print(f"[*] Spectral comparison plot saved to: {out_dir / 'evaluation_spectral.png'}")

    # ─── 5. Plot 2: Aggregate PSD Across Dataset ───
    all_psd_gt = np.array(all_psd_gt)
    all_psd_pred = np.array(all_psd_pred)
    
    psd_gt_db_all = 10.0 * np.log10(np.maximum(all_psd_gt, 1e-12))
    psd_pred_db_all = 10.0 * np.log10(np.maximum(all_psd_pred, 1e-12))
    
    mean_gt = np.mean(psd_gt_db_all, axis=0)
    std_gt = np.std(psd_gt_db_all, axis=0)
    mean_pred = np.mean(psd_pred_db_all, axis=0)
    std_pred = np.std(psd_pred_db_all, axis=0)
    
    plt.figure(figsize=(12, 6))
    plt.axvspan(50, 300, color='green', alpha=0.15, label='Tactile-Sensitive Band (50-300 Hz)')
    plt.plot(psd_freqs, mean_gt, color='blue', label='Ground Truth (Mean)', linewidth=2)
    plt.fill_between(psd_freqs, mean_gt - std_gt, mean_gt + std_gt, color='blue', alpha=0.2)
    plt.plot(psd_freqs, mean_pred, color='orange', label='Predicted (Mean)', linewidth=2)
    plt.fill_between(psd_freqs, mean_pred - std_pred, mean_pred + std_pred, color='orange', alpha=0.2)
    
    plt.title(f"Sound2Hap: Aggregate Power Spectral Density (N={len(all_results)} Validation Samples)", fontsize=14)
    plt.xlabel("Frequency (Hz) [Log Scale]", fontsize=12)
    plt.ylabel("Power Spectral Density (dB/Hz)", fontsize=12)
    plt.xscale('log')
    plt.xlim(20, 2000)
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "evaluation_aggregate_psd.png", dpi=300)
    plt.close()
    print(f"[*] Aggregate PSD plot saved to: {out_dir / 'evaluation_aggregate_psd.png'}")

    # ─── 6. Plot 3: Multi-Scale Envelope Comparison ───
    fig, axes = plt.subplots(n_plots, 3, figsize=(15, 2.5 * n_plots), sharex='col')
    if n_plots == 1:
        axes = [axes]
    fig.suptitle("Sound2Hap: Multi-Scale RMS Temporal Envelopes (64, 256, 1024 frames)", fontsize=16)

    for idx, (row_axes, s) in enumerate(zip(axes, sample_plots_data)):
        for col_idx, fl in enumerate([64, 256, 1024]):
            ax = row_axes[col_idx]
            hl = fl // 2
            env_gt = compute_rms_envelope(s['gt'], frame_len=fl, hop_len=hl)
            env_pr = compute_rms_envelope(s['pred'], frame_len=fl, hop_len=hl)
            t_axis = np.linspace(0, 1.0, num=len(env_gt))
            
            ax.plot(t_axis, env_gt, color='blue', label='GT' if idx == 0 else "", alpha=0.85)
            ax.plot(t_axis, env_pr, color='orange', label='Pred' if idx == 0 else "", alpha=0.85)
            
            corr = s['metrics'].get(f'env_corr_{fl}', 0.0)
            ax.set_title(f"Window: {fl} | Corr: {corr:.3f}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{s['class']}\nRMS Env")
            if idx == n_plots - 1:
                ax.set_xlabel("Time (s)")
            ax.grid(True, linestyle=':', alpha=0.6)
            
    axes[0][0].legend()
    plt.tight_layout()
    plt.savefig(out_dir / "evaluation_envelopes.png", dpi=300)
    plt.close()
    print(f"[*] Multi-scale envelope plot saved to: {out_dir / 'evaluation_envelopes.png'}")

if __name__ == "__main__":
    evaluate()

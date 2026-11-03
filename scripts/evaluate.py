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
from src.models.decoder import (
    TactileSEANetDecoder,
    TransientSEANetDecoder,
)
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_single_pair_metrics,
    compute_frechet_haptic_distance,
    export_publication_eval_report,
    compute_rms_envelope,
    find_optimal_micro_shift,
)

CLASS_MAP = {
    "C1": "Wood",
    "C2": "Paper/Fabric",
    "C3": "Plastic/Rubber",
    "C4": "Foam",
    "C5": "Carpet/Leather",
    "C6": "Stone/Ceramic",
    "C7": "Metal",
    "C8": "Glass",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publication-Grade Evaluation for Controlled-Tap Audio-to-Haptics"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "best_sean_decoder.pt"),
        help="Path to trained decoder checkpoint",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="Directory with processed .npy pairs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "results"),
        help="Directory to save evaluation reports and plots",
    )
    parser.add_argument(
        "--legacy_output_dir",
        type=str,
        default=str(PROJECT_ROOT / "evaluation_results"),
        help="Legacy output directory for backward compatibility",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (default: evaluate all)",
    )
    parser.add_argument(
        "--num_plot_samples",
        type=int,
        default=6,
        help="Number of individual samples to plot in figures",
    )
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device, working_sr: int, window_size: int):
    """
    Dynamically instantiates and loads either TransientSEANetDecoder or TactileSEANetDecoder
    based on the state_dict keys.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    if "output_scale" in state_dict:
        del state_dict["output_scale"]

    is_transient = any(k.startswith("base_decoder.") or k.startswith("modal_bank.") for k in state_dict.keys())
    base_decoder = TactileSEANetDecoder(input_dim=128, output_dim=1, target_length=window_size).to(device)

    if is_transient:
        decoder = TransientSEANetDecoder(
            base_decoder=base_decoder,
            latent_dim=128,
            fs_haptic=working_sr,
            num_modes=16,
        ).to(device)
        decoder.load_state_dict(state_dict, strict=False)
        model_name = "TransientSEANetDecoder (Upgraded Publication Model)"
    else:
        decoder = base_decoder
        decoder.load_state_dict(state_dict, strict=False)
        model_name = "TactileSEANetDecoder (Baseline Model)"

    decoder.eval()
    return decoder, ckpt, model_name


def evaluate():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    legacy_dir = Path(args.legacy_output_dir)
    legacy_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    # Check for working_sr in checkpoint metadata
    temp_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    working_sr = temp_ckpt.get("working_sr", 16000)
    window_size = temp_ckpt.get("window_size", 16000)

    # 1. Discover validation files (seed 42 split)
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

    print("=" * 80)
    print("CONTROLLED-TAP PUBLICATION EVALUATION SUITE")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Evaluating {len(val_pairs)} validation samples on device: {device} (Sample Rate: {working_sr} Hz)")
    print("=" * 80)

    # 2. Load Models
    encoder = FrozenAudioEncoder(target_bandwidth=1.5, device=device, input_sr=working_sr)
    decoder, ckpt, model_name = load_model_from_checkpoint(ckpt_path, device, working_sr, window_size)
    print(f"Loaded architecture: {model_name} (Epoch {ckpt.get('epoch', '?')})")

    all_results = []
    all_pub_results = []
    all_preds = []
    all_targets = []
    class_results = {c: [] for c in CLASS_MAP.keys()}

    sample_plots_data = []
    all_psd_gt = []
    all_psd_pred = []
    psd_freqs = None

    for i, (aud_path, tac_path) in enumerate(tqdm(val_pairs, desc="Evaluating")):
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

        audio_norm = audio - audio.mean()
        audio_norm = audio_norm / (np.abs(audio_norm).max() + 1e-8)
        audio_t = torch.from_numpy(audio_norm).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            latent = encoder(audio_t)
            pred_t = decoder(latent)
            if isinstance(pred_t, tuple):
                pred_t = pred_t[0]

        pred = pred_t.squeeze().cpu().numpy()
        gt = tactile

        # Compute full comprehensive metrics
        metrics = compute_all_metrics(pred, gt, sr=working_sr)
        pub_m = compute_single_pair_metrics(pred, gt, fs=working_sr, max_align_ms=1.5)

        metrics["filename"] = aud_path.name
        metrics["class"] = class_code
        metrics["class_name"] = CLASS_MAP.get(class_code, "Unknown")
        pub_m["filename"] = aud_path.name
        pub_m["class"] = class_code

        all_results.append(metrics)
        all_pub_results.append(pub_m)
        all_preds.append(pred)
        all_targets.append(gt)

        if class_code in class_results:
            class_results[class_code].append(metrics)

        # PSD computation
        nperseg = min(1024, len(pred))
        freqs, psd_gt = signal.welch(gt, fs=working_sr, nperseg=nperseg, noverlap=nperseg // 2)
        _, psd_p = signal.welch(pred, fs=working_sr, nperseg=nperseg, noverlap=nperseg // 2)
        if psd_freqs is None:
            psd_freqs = freqs
        all_psd_gt.append(psd_gt)
        all_psd_pred.append(psd_p)

        if len(sample_plots_data) < args.num_plot_samples:
            max_s = int(round(1.5 * 0.001 * working_sr))
            pred_al = find_optimal_micro_shift(pred, gt, max_shift=max_s)
            sample_plots_data.append({
                "name": stem,
                "class": CLASS_MAP.get(class_code, class_code),
                "pred": pred,
                "pred_aligned": pred_al,
                "gt": gt,
                "psd_gt": psd_gt,
                "psd_pred": psd_p,
                "freqs": freqs,
                "metrics": metrics,
                "pub_m": pub_m,
            })

    # ─── 3. Compute Fréchet Haptic Distance (FHD) & Aggregate Metrics ───
    fhd = compute_frechet_haptic_distance(all_preds, all_targets, fs=working_sr)

    # Export publication report directly to results/
    pub_summary = export_publication_eval_report(
        all_preds, all_targets, fs=working_sr, output_dir=str(out_dir)
    )

    # Also build per-class summary
    per_class_summary = {}
    for c_code, c_name in CLASS_MAP.items():
        c_items = class_results.get(c_code, [])
        if not c_items:
            continue
        per_class_summary[f"{c_code} ({c_name})"] = {
            "count": len(c_items),
            "pcc_aligned": float(np.mean([x.get("pearson_r_aligned", x["pearson_correlation"]) for x in c_items])),
            "pcc_raw": float(np.mean([x["pearson_correlation"] for x in c_items])),
            "rmse": float(np.mean([x["rmse"] for x in c_items])),
            "snr_db": float(np.mean([x.get("snr_db_aligned", x["snr_db"]) for x in c_items])),
            "peak_ratio": float(np.mean([x["peak_amplitude_ratio"] for x in c_items])),
            "envelope_r": float(np.mean([x.get("envelope_pearson_r", x.get("env_corr_256", 0.0)) for x in c_items])),
            "lsd_tactile": float(np.mean([x["lsd_tactile_band"] for x in c_items])),
            "pc_lsd_db": float(np.mean([x.get("pacinian_pc_band_lsd_db", x["lsd_tactile_band"]) for x in c_items])),
            "edc_rmse_db": float(np.mean([x.get("schroeder_edc_rmse_db", 0.0) for x in c_items])),
            "psd_cosine_sim": float(np.mean([x["psd_cosine_similarity"] for x in c_items])),
        }

    # ─── 4. Print Summary Tables ───
    print("\n" + "=" * 90)
    print("                    CONTROLLED-TAP PUBLICATION EVALUATION SUMMARY")
    print("=" * 90)
    print(f"{'Category':<16} | {'Metric':<34} | {'Mean ± Std':<22} | {'Ideal':<8}")
    print("-" * 90)

    display_metrics = [
        ("Time Domain", "Pearson Corr (Aligned ±1.5ms)", "pearson_r_aligned", "-> 1.0"),
        ("Time Domain", "Pearson Corr (Raw zero-lag)", "pearson_r_raw", "-> 1.0"),
        ("Time Domain", "RMSE (Aligned)", "waveform_rmse_aligned", "-> 0.0"),
        ("Time Domain", "MAE (Aligned)", "waveform_mae_aligned", "-> 0.0"),
        ("Time Domain", "SNR (dB, Aligned)", "snr_db_aligned", "Higher"),
        ("Time Domain", "SI-SDR (dB, Aligned)", "si_sdr_db_aligned", "Higher"),
        ("Transient Physics", "Envelope Pearson Corr", "envelope_pearson_r", "-> 1.0"),
        ("Transient Physics", "Envelope RMSE", "envelope_rmse", "-> 0.0"),
        ("Transient Physics", "Peak Amplitude Rel Error", "peak_amplitude_rel_err", "-> 0.0"),
        ("Transient Physics", "Crest Factor Error (dB)", "crest_factor_err_db", "-> 0.0"),
        ("Transient Physics", "Attack Slope Rel Error", "attack_slope_rel_err", "-> 0.0"),
        ("Transient Physics", "Schroeder EDC RMSE (dB)", "schroeder_edc_rmse_db", "-> 0.0"),
        ("Spectral & Modal", "Log Spectral Dist (LSD dB)", "log_spectral_distance_db", "-> 0.0"),
        ("Spectral & Modal", "Spectral Centroid Delta (Hz)", "spectral_centroid_err_hz", "-> 0.0"),
        ("Spectral & Modal", "PSD Jensen-Shannon Div (bits)", "psd_jsd_bits", "-> 0.0"),
        ("Psychophysical", "Pacinian (PC) Band LSD (dB)", "pacinian_pc_band_lsd_db", "-> 0.0"),
        ("Psychophysical", "Meissner (RA) Band LSD (dB)", "meissner_ra_band_lsd_db", "-> 0.0"),
    ]

    for cat, name, key, ideal in display_metrics:
        m_val = pub_summary.get(f"{key}_mean", 0.0)
        s_val = pub_summary.get(f"{key}_std", 0.0)
        print(f"{cat:<16} | {name:<34} | {m_val:7.4f} ± {s_val:6.4f}       | {ideal:<8}")

    print(f"{'Distributional':<16} | {'Fréchet Haptic Distance (FHD)':<34} | {fhd:7.4f}                | {'-> 0.0':<8}")

    print("\n" + "=" * 90)
    print("                    PER-CLASS PERFORMANCE BREAKDOWN")
    print("=" * 90)
    print(f"{'Class':<20} | {'Count':<5} | {'PCC(al)':<8} | {'Env(r)':<7} | {'SNR(dB)':<7} | {'PeakR':<6} | {'PC-LSD':<7} | {'EDC(dB)':<7} | {'CosSim':<6}")
    print("-" * 90)
    for c_label, d in per_class_summary.items():
        print(
            f"{c_label:<20} | {d['count']:<5} | {d['pcc_aligned']:<8.3f} | {d['envelope_r']:<7.3f} | "
            f"{d['snr_db']:<7.2f} | {d['peak_ratio']:<6.2f} | {d['pc_lsd_db']:<7.2f} | {d['edc_rmse_db']:<7.2f} | {d['psd_cosine_sim']:<6.3f}"
        )
    print("=" * 90 + "\n")

    # ─── 5. Save Comprehensive Reports ───
    full_report = {
        "checkpoint": ckpt_path.name,
        "model_architecture": model_name,
        "num_samples_evaluated": len(all_results),
        "working_sr": working_sr,
        "frechet_haptic_distance": fhd,
        "publication_summary": pub_summary,
        "per_class_summary": per_class_summary,
        "per_sample_results": all_results,
    }

    # Save to both out_dir and legacy_dir
    for target_dir in (out_dir, legacy_dir):
        with open(target_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)

    print(f"[*] Evaluation reports saved to:\n    - {out_dir / 'eval_metrics_report.json'}\n    - {out_dir / 'eval_summary.md'}\n    - {out_dir / 'evaluation_report.json'}")

    # ─── 6. Visualization 1: Spectral PSD Comparison with PC & RA Bands ───
    n_plots = len(sample_plots_data)
    fig, axes = plt.subplots(n_plots, 1, figsize=(13, 3.2 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    fig.suptitle("Controlled-Tap: Power Spectral Density with Mechanoreceptor Bands (Meissner RA & Pacinian PC)", fontsize=14)

    for idx, (ax, s) in enumerate(zip(axes, sample_plots_data)):
        freqs = s["freqs"]
        psd_gt_db = 10.0 * np.log10(np.maximum(s["psd_gt"], 1e-12))
        psd_pred_db = 10.0 * np.log10(np.maximum(s["psd_pred"], 1e-12))

        # Highlight bands
        ax.axvspan(5, 100, color="orange", alpha=0.12, label="Meissner RA (5-100 Hz)" if idx == 0 else "")
        ax.axvspan(40, 800, color="green", alpha=0.12, label="Pacinian PC (40-800 Hz)" if idx == 0 else "")

        ax.plot(freqs, psd_gt_db, color="blue", label="Ground Truth" if idx == 0 else "", alpha=0.85, linewidth=1.5)
        ax.plot(freqs, psd_pred_db, color="crimson", label="Predicted" if idx == 0 else "", alpha=0.85, linewidth=1.5)

        pcc_al = s["pub_m"]["pearson_r_aligned"]
        env_r = s["pub_m"]["envelope_pearson_r"]
        pc_lsd = s["pub_m"]["pacinian_pc_band_lsd_db"]
        edc = s["pub_m"]["schroeder_edc_rmse_db"]
        ax.set_title(
            f"Sample: {s['name']} [{s['class']}] | Aligned PCC: {pcc_al:.3f} | Env-r: {env_r:.3f} | "
            f"PC-LSD: {pc_lsd:.2f} dB | EDC RMSE: {edc:.2f} dB",
            fontsize=10,
        )
        ax.set_ylabel("PSD (dB/Hz)")
        ax.set_xlim(5, 2000)
        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)

    axes[0].legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Frequency (Hz) [Log Scale]")
    plt.tight_layout()
    for d in (out_dir, legacy_dir):
        plt.savefig(d / "evaluation_spectral.png", dpi=300)
    plt.close()

    # ─── 7. Visualization 2: Aggregate PSD Across Dataset ───
    all_psd_gt = np.array(all_psd_gt)
    all_psd_pred = np.array(all_psd_pred)
    psd_gt_db_all = 10.0 * np.log10(np.maximum(all_psd_gt, 1e-12))
    psd_pred_db_all = 10.0 * np.log10(np.maximum(all_psd_pred, 1e-12))

    mean_gt = np.mean(psd_gt_db_all, axis=0)
    std_gt = np.std(psd_gt_db_all, axis=0)
    mean_pred = np.mean(psd_pred_db_all, axis=0)
    std_pred = np.std(psd_pred_db_all, axis=0)

    plt.figure(figsize=(12, 6))
    plt.axvspan(40, 800, color="green", alpha=0.12, label="Pacinian PC Channel (40-800 Hz)")
    plt.plot(psd_freqs, mean_gt, color="blue", label="Ground Truth (Mean)", linewidth=2)
    plt.fill_between(psd_freqs, mean_gt - std_gt, mean_gt + std_gt, color="blue", alpha=0.2)
    plt.plot(psd_freqs, mean_pred, color="crimson", label="Predicted (Mean)", linewidth=2)
    plt.fill_between(psd_freqs, mean_pred - std_pred, mean_pred + std_pred, color="crimson", alpha=0.2)

    plt.title(f"Controlled-Tap: Aggregate Power Spectral Density (N={len(all_results)} Samples | FHD: {fhd:.3f})", fontsize=13)
    plt.xlabel("Frequency (Hz) [Log Scale]", fontsize=11)
    plt.ylabel("Power Spectral Density (dB/Hz)", fontsize=11)
    plt.xscale("log")
    plt.xlim(10, 2000)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    for d in (out_dir, legacy_dir):
        plt.savefig(d / "evaluation_aggregate_psd.png", dpi=300)
    plt.close()

    # ─── 8. Visualization 3: Waveform Alignment & Schroeder EDC Decay Curves ───
    fig, axes = plt.subplots(n_plots, 2, figsize=(14, 2.7 * n_plots))
    if n_plots == 1:
        axes = np.expand_dims(axes, 0)
    fig.suptitle("Controlled-Tap: Time-Domain Waveforms (Left) & Schroeder Energy Decay Curves (Right)", fontsize=14)

    for idx, (row, s) in enumerate(zip(axes, sample_plots_data)):
        t_ms = np.linspace(0, 1000.0 * (len(s["gt"]) / float(working_sr)), len(s["gt"]))
        # Focus on first 150 ms for tap contact impact
        mask_150 = t_ms <= 150.0

        # Left: Waveform
        ax_w = row[0]
        ax_w.plot(t_ms[mask_150], s["gt"][mask_150], color="blue", label="GT" if idx == 0 else "", alpha=0.7)
        ax_w.plot(t_ms[mask_150], s["pred_aligned"][mask_150], color="crimson", label="Pred (Aligned)" if idx == 0 else "", alpha=0.8)
        pcc = s["pub_m"]["pearson_r_aligned"]
        ax_w.set_title(f"{s['name']} [{s['class']}] | Aligned PCC: {pcc:.3f}", fontsize=10)
        ax_w.set_ylabel("Norm G-Force")
        if idx == n_plots - 1:
            ax_w.set_xlabel("Time (ms)")
        ax_w.grid(True, linestyle=":", alpha=0.6)

        # Right: Schroeder EDC
        ax_e = row[1]
        edc_gt = np.cumsum((s["gt"] ** 2)[::-1])[::-1]
        edc_pr = np.cumsum((s["pred_aligned"] ** 2)[::-1])[::-1]
        edc_gt_db = 10.0 * np.log10(np.clip(edc_gt / (edc_gt[0] + 1e-8), 1e-4, 1.0))
        edc_pr_db = 10.0 * np.log10(np.clip(edc_pr / (edc_pr[0] + 1e-8), 1e-4, 1.0))

        ax_e.plot(t_ms[mask_150], edc_gt_db[mask_150], color="blue", label="GT EDC" if idx == 0 else "", linewidth=1.5)
        ax_e.plot(t_ms[mask_150], edc_pr_db[mask_150], color="crimson", label="Pred EDC" if idx == 0 else "", linewidth=1.5)
        edc_rmse = s["pub_m"]["schroeder_edc_rmse_db"]
        ax_e.set_title(f"Schroeder EDC Decay | RMSE: {edc_rmse:.2f} dB", fontsize=10)
        ax_e.set_ylabel("Decay (dB)")
        ax_e.set_ylim(-35, 2)
        if idx == n_plots - 1:
            ax_e.set_xlabel("Time (ms)")
        ax_e.grid(True, linestyle=":", alpha=0.6)

    axes[0, 0].legend(loc="upper right")
    axes[0, 1].legend(loc="upper right")
    plt.tight_layout()
    for d in (out_dir, legacy_dir):
        plt.savefig(d / "evaluation_waveforms_and_edc.png", dpi=300)
    plt.close()

    print(f"[*] Visualizations saved successfully to:\n    - {out_dir / 'evaluation_spectral.png'}\n    - {out_dir / 'evaluation_aggregate_psd.png'}\n    - {out_dir / 'evaluation_waveforms_and_edc.png'}")


if __name__ == "__main__":
    evaluate()

import sys
import torch
import random
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt

# Inject project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_dataloaders
from src.models.encodec_wrapper import FrozenAudioEncoder
from src.models.decoder import (
    TactileSEANetDecoder,
    TransientSEANetDecoder,
)
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_single_pair_metrics,
    find_optimal_micro_shift,
)


def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = PROJECT_ROOT / "data" / "processed"
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "best_sean_decoder.pt"
    
    # Check v2 checkpoint first, fall back to best_sean_decoder.pt
    v2_path = PROJECT_ROOT / "checkpoints" / "best_sean_decoder_v2.pt"
    if v2_path.exists():
        checkpoint_path = v2_path
    
    if not checkpoint_path.exists():
        print(f"Error: Could not find checkpoint at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    working_sr = checkpoint.get("working_sr", 16000)
    window_size = checkpoint.get("window_size", 16000)

    print(f"Loaded checkpoint from: {checkpoint_path.name}")
    print("1. Loading Validation Data...")
    _, val_loader = get_dataloaders(processed_dir, batch_size=1, window_size=window_size)
    if val_loader is None:
        print("Error: No validation data available.")
        return

    val_samples = list(val_loader)
    audio_tensor, true_tactile_tensor = random.choice(val_samples)
    audio_tensor = audio_tensor.to(device)
    out_channels = true_tactile_tensor.shape[1]

    print("2. Initializing Models and Loading Weights...")
    encoder = FrozenAudioEncoder(target_bandwidth=1.5, device=device, input_sr=working_sr)
    
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if "output_scale" in state_dict:
        del state_dict["output_scale"]

    is_transient = any(k.startswith("base_decoder.") or k.startswith("modal_bank.") for k in state_dict.keys())
    base_decoder = TactileSEANetDecoder(input_dim=128, output_dim=out_channels, target_length=window_size).to(device)

    if is_transient:
        decoder = TransientSEANetDecoder(
            base_decoder=base_decoder,
            latent_dim=128,
            fs_haptic=working_sr,
            num_modes=16,
        ).to(device)
        decoder.load_state_dict(state_dict, strict=False)
        arch_name = "TransientSEANetDecoder (Upgraded Model)"
    else:
        decoder = base_decoder
        decoder.load_state_dict(state_dict, strict=False)
        arch_name = "TactileSEANetDecoder (Baseline)"

    decoder.eval()
    print(f"Active Architecture: {arch_name}")

    print("3. Generating Tactile Waveform...")
    with torch.no_grad():
        latent_embeddings = encoder(audio_tensor)
        predicted_tactile_tensor = decoder(latent_embeddings)
        if isinstance(predicted_tactile_tensor, tuple):
            predicted_tactile_tensor = predicted_tactile_tensor[0]

    audio_np = audio_tensor.squeeze().cpu().numpy()
    true_tactile_np = true_tactile_tensor.squeeze().cpu().numpy()
    pred_tactile_np = predicted_tactile_tensor.squeeze().cpu().numpy()

    if true_tactile_np.ndim > 1:
        true_tactile_np = true_tactile_np[0]
        pred_tactile_np = pred_tactile_np[0]

    # Post-generation tactile bandpass filter (gentle 2nd-order butterworth 50-300Hz)
    sos = butter(2, [50, 300], btype="bandpass", fs=working_sr, output="sos")
    pred_tactile_filtered = sosfiltfilt(sos, pred_tactile_np).astype(np.float32)

    # Compute comprehensive publication metrics
    max_s = int(round(1.5 * 0.001 * working_sr))
    pred_aligned = find_optimal_micro_shift(pred_tactile_np, true_tactile_np, max_shift=max_s)
    pub_metrics = compute_single_pair_metrics(pred_tactile_np, true_tactile_np, fs=working_sr)
    all_metrics = compute_all_metrics(pred_tactile_np, true_tactile_np, sr=working_sr)

    print("\n" + "=" * 65)
    print("           INFERENCE QUANTITATIVE EVALUATION")
    print("=" * 65)
    print(f"Pearson Correlation (Raw):     {pub_metrics['pearson_r_raw']:.4f}")
    print(f"Pearson Correlation (Aligned): {pub_metrics['pearson_r_aligned']:.4f}")
    print(f"Envelope Pearson Correlation:  {pub_metrics['envelope_pearson_r']:.4f}")
    print(f"Waveform RMSE (Aligned):       {pub_metrics['waveform_rmse_aligned']:.4f}")
    print(f"Signal-to-Noise (SNR Aligned): {pub_metrics['snr_db_aligned']:.2f} dB")
    print(f"SI-SDR (Aligned):              {pub_metrics['si_sdr_db_aligned']:.2f} dB")
    print(f"Peak Amplitude Relative Err:   {pub_metrics['peak_amplitude_rel_err']:.2%}")
    print(f"Crest Factor Error (dB):       {pub_metrics['crest_factor_err_db']:.2f} dB")
    print(f"Schroeder EDC RMSE:            {pub_metrics['schroeder_edc_rmse_db']:.2f} dB")
    print(f"Log Spectral Distance (LSD):   {pub_metrics['log_spectral_distance_db']:.2f} dB")
    print(f"Pacinian PC Band (40-800Hz):   {pub_metrics['pacinian_pc_band_lsd_db']:.2f} dB")
    print(f"Meissner RA Band (5-100Hz):    {pub_metrics['meissner_ra_band_lsd_db']:.2f} dB")
    print(f"PSD Jensen-Shannon Divergence: {pub_metrics['psd_jsd_bits']:.4f} bits")
    print("=" * 65 + "\n")

    time_axis = np.linspace(0, window_size / working_sr, num=window_size)

    print("4. Plotting Results...")
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    title_str = (
        f"Sound2Hap Controlled Tap: Inference Verification [{arch_name}]\n"
        f"Aligned PCC: {pub_metrics['pearson_r_aligned']:.3f} (Raw: {pub_metrics['pearson_r_raw']:.3f}) | "
        f"Env-r: {pub_metrics['envelope_pearson_r']:.3f} | SNR: {pub_metrics['snr_db_aligned']:.1f} dB | "
        f"PC-LSD: {pub_metrics['pacinian_pc_band_lsd_db']:.1f} dB | EDC RMSE: {pub_metrics['schroeder_edc_rmse_db']:.1f} dB"
    )
    fig.suptitle(title_str, fontsize=12)

    axes[0].plot(time_axis, audio_np, color="gray", alpha=0.8)
    axes[0].set_title("Input Audio Modality (soundTap @ 16kHz)")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(time_axis, true_tactile_np, color="blue", alpha=0.8)
    axes[1].set_title(f"Ground Truth Tactile Vibration (Peak: {np.max(np.abs(true_tactile_np)):.2f}, RMS: {np.sqrt(np.mean(true_tactile_np**2)):.2f})")
    axes[1].set_ylabel("Norm G-Force")

    axes[2].plot(time_axis, pred_tactile_np, color="orange", alpha=0.8, label="Raw")
    axes[2].plot(time_axis, pred_aligned, color="red", linestyle="--", alpha=0.7, label="Micro-Aligned (±1.5ms)")
    axes[2].set_title("Model Predicted Tactile Vibration (Raw & Micro-Aligned)")
    axes[2].set_ylabel("Norm G-Force")
    axes[2].legend(loc="upper right")

    axes[3].plot(time_axis, pred_tactile_filtered, color="crimson", alpha=0.8)
    axes[3].set_title("Post-Filtered Tactile Vibration (50-300Hz Sensitive Range)")
    axes[3].set_ylabel("Norm G-Force")
    axes[3].set_xlabel("Time (Seconds)")

    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_img = PROJECT_ROOT / "inference_result.png"
    plt.savefig(output_img, dpi=300)
    print(f"\n[*] Verification plot saved successfully to: {output_img.name}")


if __name__ == "__main__":
    run_inference()

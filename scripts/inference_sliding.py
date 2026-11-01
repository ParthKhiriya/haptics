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

from src.data.dataset_sliding import get_sliding_dataloaders
from src.models.encodec_wrapper import FrozenAudioEncoder
from src.models.decoder_sliding import SlidingSEANetDecoder, SlidingTactileDecoder
from src.evaluation.metrics import compute_all_metrics

def run_inference_sliding():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = PROJECT_ROOT / "data" / "processed_sliding"
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "best_sliding_decoder.pt"
    
    if not checkpoint_path.exists():
        print(f"Error: Could not find checkpoint at {checkpoint_path}")
        print("Please train the model first by running scripts/train_sliding.py")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    working_sr = checkpoint.get('working_sr', 16000)
    audio_length = checkpoint.get('audio_length', 16000)
    haptic_length = checkpoint.get('haptic_length', 4800)

    print("1. Loading Sliding Validation Data...")
    _, val_loader = get_sliding_dataloaders(
        processed_dir, batch_size=1, 
        audio_length=audio_length, 
        haptic_length=haptic_length,
        seed=42,
    )
    if val_loader is None or len(val_loader) == 0:
        print("Error: No validation data available.")
        return
        
    val_samples = list(val_loader)
    audio_tensor, true_tactile_tensor = random.choice(val_samples)
    
    audio_tensor = audio_tensor.to(device)
    out_channels = true_tactile_tensor.shape[1]

    print("2. Initializing Models and Loading Weights...")
    encoder = FrozenAudioEncoder(target_bandwidth=1.5, device=device, input_sr=working_sr)
    decoder = SlidingSEANetDecoder(
        latent_dim=128, 
        output_dim=out_channels, 
        target_length=haptic_length,
        num_tactile_bands=40
    ).to(device)
    
    state_dict = checkpoint['model_state_dict']
    decoder.load_state_dict(state_dict, strict=True)
    decoder.eval()

    print("3. Generating Sliding Tactile Waveform...")
    with torch.no_grad():
        latent_embeddings = encoder(audio_tensor)
        predicted_tactile_tensor = decoder(latent_embeddings, audio=audio_tensor)

    audio_np = audio_tensor.squeeze().cpu().numpy()
    true_tactile_np = true_tactile_tensor.squeeze().cpu().numpy()
    pred_tactile_np = predicted_tactile_tensor.squeeze().cpu().numpy()

    if true_tactile_np.ndim > 1:
        true_tactile_np = true_tactile_np[0]
        pred_tactile_np = pred_tactile_np[0]

    # Post-generation tactile bandpass filter (2nd-order butterworth 30-800Hz for sliding)
    sos = butter(2, [30, 800], btype='bandpass', fs=working_sr, output='sos')
    pred_tactile_filtered = sosfiltfilt(sos, pred_tactile_np).astype(np.float32)

    # Compute comprehensive evaluation metrics with tactile band set to 30-800 Hz
    raw_metrics = compute_all_metrics(pred_tactile_np, true_tactile_np, sr=working_sr,
                                      tactile_low=30.0, tactile_high=800.0)
    filt_metrics = compute_all_metrics(pred_tactile_filtered, true_tactile_np, sr=working_sr,
                                       tactile_low=30.0, tactile_high=800.0)

    print("\n" + "="*60)
    print("       CONTROLLED SLIDING INFERENCE EVALUATION (Phase 2.3)")
    print("="*60)
    print(f"Pearson Correlation (PCC): {raw_metrics['pearson_correlation']:.4f} (Filtered: {filt_metrics['pearson_correlation']:.4f})")
    print(f"RMSE:                     {raw_metrics['rmse']:.4f} (Filtered: {filt_metrics['rmse']:.4f})")
    print(f"Signal-to-Noise (SNR dB): {raw_metrics['snr_db']:.2f} dB (Filtered: {filt_metrics['snr_db']:.2f} dB)")
    print(f"Peak Amplitude Ratio:     {raw_metrics['peak_amplitude_ratio']:.3f}x [GT: {raw_metrics['peak_gt']:.2f}, Pred: {raw_metrics['peak_pred']:.2f}]")
    print(f"RMS Energy Ratio:         {raw_metrics['rms_ratio']:.3f}x [GT: {raw_metrics['rms_gt']:.2f}, Pred: {raw_metrics['rms_pred']:.2f}]")
    print(f"Tactile Band (30-800Hz) LSD: {raw_metrics['lsd_tactile_band']:.2f} dB")
    print(f"Tactile Energy Ratio:     {raw_metrics['tactile_band_energy_ratio']:.3f}x")
    print(f"PSD Cosine Similarity:    {raw_metrics['psd_cosine_similarity']:.4f}")
    print(f"Envelope Corr (256):      {raw_metrics.get('env_corr_256', 0.0):.4f}")
    print(f"Envelope Corr (1024):     {raw_metrics.get('env_corr_1024', 0.0):.4f}")
    print(f"Spectral Centroid Delta:  {raw_metrics['spectral_centroid_delta_hz']:.1f} Hz (GT: {raw_metrics['spectral_centroid_gt_hz']:.1f} Hz, Pred: {raw_metrics['spectral_centroid_pred_hz']:.1f} Hz)")
    print(f"SPQI-Lite:                {raw_metrics['spqi_lite']:.3f}")
    print("="*60 + "\n")

    audio_time = np.linspace(0, audio_length / working_sr, num=audio_length)
    haptic_time = np.linspace(0, haptic_length / working_sr, num=haptic_length)

    print("4. Plotting Results...")
    fig, axes = plt.subplots(4, 1, figsize=(13, 11))
    title_str = (
        f"Controlled Sliding (Phase 2.3): Audio-to-Haptic Cross-Modal Synthesis\n"
        f"Tactile LSD: {raw_metrics['lsd_tactile_band']:.1f} dB | PSD CosSim: {raw_metrics['psd_cosine_similarity']:.3f} | "
        f"Peak Ratio: {raw_metrics['peak_amplitude_ratio']:.2f}x | RMS Ratio: {raw_metrics['rms_ratio']:.2f}x | "
        f"Centroid Delta: {raw_metrics['spectral_centroid_delta_hz']:.1f} Hz"
    )
    fig.suptitle(title_str, fontsize=13)

    axes[0].plot(audio_time, audio_np, color='gray', alpha=0.8)
    axes[0].set_title(f"Input Sliding Audio Modality (sound @ 16kHz, Duration: {audio_length/working_sr:.1f}s, Aligned 0.3s window in [:4800])")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlabel("Time (Seconds)")

    axes[1].plot(haptic_time, true_tactile_np, color='blue', alpha=0.8)
    axes[1].set_title(f"Ground Truth accelDFT Tactile Vibration (0.3s, Peak: {raw_metrics['peak_gt']:.2f}, RMS: {raw_metrics['rms_gt']:.2f}, Centroid: {raw_metrics['spectral_centroid_gt_hz']:.1f} Hz)")
    axes[1].set_ylabel("Norm G-Force")
    axes[1].set_xlabel("Time (Seconds)")

    axes[2].plot(haptic_time, pred_tactile_np, color='orange', alpha=0.8)
    axes[2].set_title(f"Model Predicted Sliding Tactile - Raw Output (Peak: {raw_metrics['peak_pred']:.2f}, RMS: {raw_metrics['rms_pred']:.2f}, Centroid: {raw_metrics['spectral_centroid_pred_hz']:.1f} Hz)")
    axes[2].set_ylabel("Norm G-Force")
    axes[2].set_xlabel("Time (Seconds)")

    axes[3].plot(haptic_time, pred_tactile_filtered, color='red', alpha=0.8)
    axes[3].set_title("Post-Filtered Sliding Vibration (30-800Hz Sensitive Range)")
    axes[3].set_ylabel("Norm G-Force")
    axes[3].set_xlabel("Time (Seconds)")

    for ax in axes:
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    
    output_img = PROJECT_ROOT / "inference_sliding_result.png"
    plt.savefig(output_img, dpi=300)
    print(f"\n[*] Verification plot saved successfully to: {output_img.name}")

if __name__ == "__main__":
    run_inference_sliding()

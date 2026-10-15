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
from src.models.decoder import TactileSEANetDecoder

def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = PROJECT_ROOT / "data" / "processed"
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "best_sean_decoder.pt"
    
    if not checkpoint_path.exists():
        print(f"Error: Could not find checkpoint at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    working_sr = checkpoint.get('working_sr', 16000)
    window_size = checkpoint.get('window_size', 16000)

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
    decoder = TactileSEANetDecoder(input_dim=128, output_dim=out_channels, target_length=window_size).to(device)
    
    decoder.load_state_dict(checkpoint['model_state_dict'])
    decoder.eval()

    print("3. Generating Tactile Waveform...")
    with torch.no_grad():
        latent_embeddings = encoder(audio_tensor)
        predicted_tactile_tensor = decoder(latent_embeddings)

    audio_np = audio_tensor.squeeze().cpu().numpy()
    true_tactile_np = true_tactile_tensor.squeeze().cpu().numpy()
    pred_tactile_np = predicted_tactile_tensor.squeeze().cpu().numpy()

    if true_tactile_np.ndim > 1:
        true_tactile_np = true_tactile_np[0]
        pred_tactile_np = pred_tactile_np[0]

    # Post-generation tactile bandpass filter (gentle 2nd-order butterworth 50-300Hz)
    sos = butter(2, [50, 300], btype='bandpass', fs=working_sr, output='sos')
    pred_tactile_filtered = sosfiltfilt(sos, pred_tactile_np).astype(np.float32)

    time_axis = np.linspace(0, window_size / working_sr, num=window_size)

    print("4. Plotting Results...")
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Sound2Hap (Controlled Tap): Cross-Modal Inference Verification", fontsize=16)

    axes[0].plot(time_axis, audio_np, color='gray', alpha=0.8)
    axes[0].set_title("Input Audio Modality (soundTap @ 16kHz)")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(time_axis, true_tactile_np, color='blue', alpha=0.8)
    axes[1].set_title("Ground Truth Tactile Vibration (accelTap @ 16kHz)")
    axes[1].set_ylabel("Norm G-Force")

    axes[2].plot(time_axis, pred_tactile_np, color='orange', alpha=0.8)
    axes[2].set_title("Model Predicted Tactile Vibration (Raw Output)")
    axes[2].set_ylabel("Norm G-Force")

    axes[3].plot(time_axis, pred_tactile_filtered, color='red', alpha=0.8)
    axes[3].set_title("Post-Filtered Tactile Vibration (50-300Hz Sensitive Range)")
    axes[3].set_ylabel("Norm G-Force")
    axes[3].set_xlabel("Time (Seconds)")

    plt.tight_layout()
    
    output_img = PROJECT_ROOT / "inference_result.png"
    plt.savefig(output_img, dpi=300)
    print(f"\n[*] Verification plot saved successfully to: {output_img.name}")

if __name__ == "__main__":
    run_inference()
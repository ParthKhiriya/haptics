import sys
import torch
import random
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

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

    print("1. Loading Validation Data...")
    _, val_loader = get_dataloaders(processed_dir, batch_size=1, window_size=10000)
    
    val_samples = list(val_loader)
    audio_tensor, true_tactile_tensor = random.choice(val_samples)
    
    audio_tensor = audio_tensor.to(device)
    out_channels = true_tactile_tensor.shape[1]

    print("2. Initializing Models and Loading Weights...")
    encoder = FrozenAudioEncoder(target_bandwidth=1.5, device=device)
    decoder = TactileSEANetDecoder(input_dim=128, output_dim=out_channels, target_length=10000).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
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

    # --- SANE MASTERING ---
    # Only use a light noise gate to clean up true silence. NO dynamic scaling.
    noise_gate_threshold = 0.05
    pred_tactile_np[np.abs(pred_tactile_np) < noise_gate_threshold] = 0.0

    time_axis = np.linspace(0, 1.0, num=10000)

    print("4. Plotting Results...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Sound2Hap: Cross-Modal Inference Verification", fontsize=16)

    axes[0].plot(time_axis, audio_np, color='gray', alpha=0.8)
    axes[0].set_title("Input Audio Modality (Microphone)")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(time_axis, true_tactile_np, color='blue', alpha=0.8)
    axes[1].set_title("Ground Truth Tactile Vibration (Accelerometer)")
    axes[1].set_ylabel("G-Force (Norm)")

    axes[2].plot(time_axis, pred_tactile_np, color='red', alpha=0.8)
    axes[2].set_title("Model Predicted Tactile Vibration")
    axes[2].set_ylabel("G-Force (Norm)")
    axes[2].set_xlabel("Time (Seconds)")

    plt.tight_layout()
    
    output_img = PROJECT_ROOT / "inference_result.png"
    plt.savefig(output_img, dpi=300)
    print(f"\n[*] Verification plot saved successfully to: {output_img.name}")

if __name__ == "__main__":
    run_inference()
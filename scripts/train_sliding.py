import sys
import os
import torch
from pathlib import Path
from tqdm import tqdm

# Inject project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_sliding import get_sliding_dataloaders
from src.models.encodec_wrapper import FrozenAudioEncoder
from src.models.decoder_sliding import SlidingSEANetDecoder, SlidingTactileDecoder
from src.models.loss_sliding import SlidingTextureLoss, SlidingHapticLoss

def train_sliding():
    # ─── Configuration ───
    EPOCHS = 150
    BATCH_SIZE = 16            # Robust batch size for local GPU memory
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    WORKING_SR = 16000
    AUDIO_LENGTH = 16000       # 16000 samples = 1.0 second audio input (aligned 0.3s is in [:4800])
    HAPTIC_LENGTH = 4800       # 4800 samples = 0.3 second haptic target
    PATIENCE = 40              # Extended patience for Phase 2.3 fine-tuning
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = PROJECT_ROOT / "data" / "processed_sliding"
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    print(f"Starting Controlled Sliding Training (Phase 2.3 Acoustic-Guided Overhaul) on {device}...")
    print(f"Sample Rate: {WORKING_SR} Hz | Audio Window: {AUDIO_LENGTH} | Haptic Target: {HAPTIC_LENGTH} | Patience: {PATIENCE}")

    # ─── 1. Load Data (Stratified Split) ───
    train_loader, val_loader = get_sliding_dataloaders(
        processed_dir, 
        batch_size=BATCH_SIZE, 
        audio_length=AUDIO_LENGTH,
        haptic_length=HAPTIC_LENGTH,
        train_ratio=0.8,
        seed=42,
    )
    
    if train_loader is None or len(train_loader) == 0:
        print("Error: No data available. Make sure preprocessing has finished.")
        print(f"Expected processed data in: {processed_dir}")
        return

    # Dynamic target channels
    _, sample_tactile = next(iter(train_loader))
    out_channels = sample_tactile.shape[1]

    # ─── 2. Initialize Models & Optimizer ───
    encoder = FrozenAudioEncoder(target_bandwidth=1.5, device=device, input_sr=WORKING_SR)
    
    decoder = SlidingSEANetDecoder(
        latent_dim=128, 
        output_dim=out_channels, 
        target_length=HAPTIC_LENGTH,
        num_tactile_bands=40,
    ).to(device)
    
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = SlidingTextureLoss(sr=WORKING_SR).to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0

    # ─── 3. Training Loop ───
    for epoch in range(1, EPOCHS + 1):
        decoder.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train Sliding]")
        for audio, tactile in train_bar:
            audio, tactile = audio.to(device), tactile.to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                latent_embeddings = encoder(audio)
            predicted_tactile = decoder(latent_embeddings, audio=audio)
            
            loss, metrics = criterion(predicted_tactile, tactile, return_components=True)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=2.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({
                'loss': f"{loss.item():.3f}", 
                'psd': f"{metrics.get('psd', 0.0):.3f}",
                'cent': f"{metrics.get('cent', metrics.get('centroid', 0.0)):.3f}",
                'env': f"{metrics.get('env', 0.0):.3f}",
                'stat': f"{metrics.get('stat', 0.0):.3f}",
            })
            
        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()
        
        # ─── 4. Validation Loop ───
        decoder.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val Sliding]")
        
        with torch.no_grad():
            for audio, tactile in val_bar:
                audio, tactile = audio.to(device), tactile.to(device)
                
                latent_embeddings = encoder(audio)
                predicted_tactile = decoder(latent_embeddings, audio=audio)
                
                loss, _ = criterion(predicted_tactile, tactile, return_components=True)
                val_loss += loss.item()
                val_bar.set_postfix({'val_loss': f"{loss.item():.3f}"})
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch} Summary -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # ─── 5. Checkpointing & Early Stopping ───
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_path = checkpoint_dir / "best_sliding_decoder.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'working_sr': WORKING_SR,
                'audio_length': AUDIO_LENGTH,
                'haptic_length': HAPTIC_LENGTH,
            }, save_path)
            print(f"[*] New best model saved to {save_path.name} (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[*] Early stopping triggered after {PATIENCE} epochs with no improvement.")
                break

if __name__ == "__main__":
    train_sliding()

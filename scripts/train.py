import sys
import os
import torch
from pathlib import Path
from tqdm import tqdm

# Inject project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_dataloaders
from src.models.encodec_wrapper import FrozenAudioEncoder
from src.models.decoder import TactileSEANetDecoder
from src.models.loss import Sound2HapLoss

def train():
    # --- Configuration ---
    EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-4
    WINDOW_SIZE = 10000
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = PROJECT_ROOT / "data" / "processed"
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    print(f"Starting Training on {device}...")

    # --- 1. Load Data ---
    train_loader, val_loader = get_dataloaders(
        processed_dir, 
        batch_size=BATCH_SIZE, 
        window_size=WINDOW_SIZE
    )
    
    # Check target channels dynamically from the first batch
    _, sample_tactile = next(iter(train_loader))
    out_channels = sample_tactile.shape[1]

    # --- 2. Initialize Models & Loss ---
    encoder = FrozenAudioEncoder(target_bandwidth=1.5, device=device)
    
    decoder = TactileSEANetDecoder(
        input_dim=128, 
        output_dim=out_channels, 
        target_length=WINDOW_SIZE
    ).to(device)
    
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))
    criterion = Sound2HapLoss(sample_rate=10000).to(device)
    
    best_val_loss = float('inf')

    # --- 3. Training Loop ---
    for epoch in range(1, EPOCHS + 1):
        decoder.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for audio, tactile in train_bar:
            audio, tactile = audio.to(device), tactile.to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                latent_embeddings = encoder(audio)
            predicted_tactile = decoder(latent_embeddings)
            
            loss, metrics = criterion(predicted_tactile, tactile)
            
            # Backpropagation
            loss.backward()
            
            # Fix 3: Prevent gradient explosions (caps the maximum gradient at 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}", 'mse': f"{metrics['mse']:.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- 4. Validation Loop ---
        decoder.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]")
        
        with torch.no_grad():
            for audio, tactile in val_bar:
                audio, tactile = audio.to(device), tactile.to(device)
                
                latent_embeddings = encoder(audio)
                predicted_tactile = decoder(latent_embeddings)
                
                loss, metrics = criterion(predicted_tactile, tactile)
                val_loss += loss.item()
                val_bar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch} Summary -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # --- 5. Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = checkpoint_dir / "best_sean_decoder.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss
            }, save_path)
            print(f"[*] New best model saved to {save_path.name}")

if __name__ == "__main__":
    train()
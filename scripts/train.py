import sys
import os
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Inject project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_dataloaders
from src.models.encodec_wrapper import FrozenAudioEncoder
from src.models.decoder import (
    TactileSEANetDecoder,
    TransientSEANetDecoder,
    load_and_upgrade_checkpoint,
)
from src.models.loss import ControlledTapPublicationLoss
from src.models.discriminator import (
    MultiPeriodWaveformDiscriminator,
    discriminator_lsgan_loss,
    generator_adv_and_fm_loss,
    MaterialContrastiveSignatureLoss,
)
from src.evaluation.metrics import compute_single_pair_metrics


class EMAModel:
    """
    Exponential Moving Average (EMA) of generator parameters for stabilized synthesis.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.clone().detach()
            for k, v in model.state_dict().items()
            if v.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow

    def copy_to(self, model: nn.Module):
        model_dict = model.state_dict()
        for k, v in self.shadow.items():
            if k in model_dict:
                model_dict[k].copy_(v)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Controlled-Tap Audio-to-Haptics: Publication-Grade 2-Stage Training"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs (default: 100)")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size (default: 16)")
    parser.add_argument("--lr_g", type=float, default=3e-4, help="Generator learning rate (default: 3e-4)")
    parser.add_argument("--lr_d", type=float, default=1.5e-4, help="Discriminator learning rate (default: 1.5e-4)")
    parser.add_argument("--stage2_epoch", type=int, default=40, help="Epoch to enable Stage-2 adversarial training (default: 40)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (default: 20)")
    parser.add_argument("--working_sr", type=int, default=16000, help="Audio and haptic sample rate (default: 16000)")
    parser.add_argument("--no_warm_start", action="store_true", help="Disable warm-start from best_sean_decoder.pt")
    parser.add_argument("--checkpoint_dir", type=str, default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--processed_dir", type=str, default=str(PROJECT_ROOT / "data" / "processed"))
    return parser.parse_args()


def train():
    args = parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR_G = args.lr_g
    LR_D = args.lr_d
    STAGE2_EPOCH = args.stage2_epoch
    PATIENCE = args.patience
    WORKING_SR = args.working_sr
    WINDOW_SIZE = WORKING_SR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = Path(args.processed_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CONTROLLED-TAP AUDIO-TO-HAPTICS: PUBLICATION TRAINING PIPELINE")
    print(f"Device: {device} | Sample Rate: {WORKING_SR} Hz | Window: {WINDOW_SIZE} samples")
    print(f"Curriculum: Stage 1 (Epochs 1-{STAGE2_EPOCH}) -> Stage 2 (Epochs {STAGE2_EPOCH + 1}-{EPOCHS})")
    print("=" * 80)

    # ─── 1. Load Data with Auxiliary Tap Metadata ───
    train_loader, val_loader = get_dataloaders(
        processed_dir,
        batch_size=BATCH_SIZE,
        window_size=WINDOW_SIZE,
        return_aux=True,
    )

    if train_loader is None or len(train_loader) == 0:
        print(f"Error: No processed data found in {processed_dir}. Run preprocessing first.")
        return

    sample_audio, sample_tactile, _ = next(iter(train_loader))
    out_channels = sample_tactile.shape[1]

    # ─── 2. Initialize Models & Warm-Start ───
    encoder = FrozenAudioEncoder(target_bandwidth=1.5, device=device, input_sr=WORKING_SR)

    base_decoder = TactileSEANetDecoder(
        input_dim=128,
        output_dim=out_channels,
        target_length=WINDOW_SIZE,
    )

    ckpt_candidate = checkpoint_dir / "best_sean_decoder.pt"
    if not args.no_warm_start and ckpt_candidate.exists():
        print(f"[*] Warm-starting TransientSEANetDecoder from: {ckpt_candidate.name}")
        decoder = load_and_upgrade_checkpoint(
            base_decoder=base_decoder,
            checkpoint_path=str(ckpt_candidate),
            latent_dim=128,
            fs_haptic=WORKING_SR,
            num_modes=16,
            device=str(device),
        )
    else:
        print("[*] Initializing fresh TransientSEANetDecoder (no warm-start)...")
        decoder = TransientSEANetDecoder(
            base_decoder=base_decoder,
            latent_dim=128,
            fs_haptic=WORKING_SR,
            num_modes=16,
        ).to(device)

    # Maintain Exponential Moving Average of generator
    ema = EMAModel(decoder, decay=0.999)

    # Generator Optimizer & Scheduler
    optimizer_g = torch.optim.AdamW(
        decoder.parameters(),
        lr=LR_G,
        betas=(0.8, 0.99),
        weight_decay=1e-4,
    )
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_g, T_max=EPOCHS, eta_min=1e-5
    )

    # Multi-Period Discriminator (MPD) for Stage 2
    discriminator = MultiPeriodWaveformDiscriminator(periods=(2, 3, 5, 7, 11)).to(device)
    optimizer_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=LR_D,
        betas=(0.8, 0.99),
        weight_decay=1e-4,
    )
    stage2_epochs = max(1, EPOCHS - STAGE2_EPOCH)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_d, T_max=stage2_epochs, eta_min=1e-5
    )

    # Losses
    pub_criterion = ControlledTapPublicationLoss(fs=WORKING_SR, max_shift_samples=6).to(device)
    contrastive_criterion = MaterialContrastiveSignatureLoss(temperature=0.1).to(device)

    best_val_score = -float("inf")
    patience_counter = 0

    # ─── 3. Training Loop ───
    for epoch in range(1, EPOCHS + 1):
        in_stage_2 = epoch > STAGE2_EPOCH
        stage_name = "Stage 2: Adversarial MPD" if in_stage_2 else "Stage 1: Physics/Spectral Warmup"

        decoder.train()
        if in_stage_2:
            discriminator.train()

        train_loss_g_total = 0.0
        train_loss_d_total = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [{stage_name}]")
        for audio, tactile, aux in train_bar:
            audio = audio.to(device)
            tactile = tactile.to(device)

            with torch.no_grad():
                latent = encoder(audio)

            # -----------------
            # Discriminator Step (Stage 2 only)
            # -----------------
            if in_stage_2:
                optimizer_d.zero_grad()
                with torch.no_grad():
                    pred_tactile_d, _ = decoder(latent, return_aux=True)

                y_d_rs, y_d_gs, _, _ = discriminator(tactile, pred_tactile_d.detach())
                loss_d = discriminator_lsgan_loss(y_d_rs, y_d_gs)

                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=2.0)
                optimizer_d.step()
                train_loss_d_total += loss_d.item()

            # -----------------
            # Generator Step
            # -----------------
            optimizer_g.zero_grad()
            pred_tactile, dec_aux = decoder(latent, return_aux=True)

            loss_pub, pub_metrics = pub_criterion(pred_tactile, tactile, aux=dec_aux)
            loss_contrast = contrastive_criterion(pred_tactile, tactile)

            loss_g = loss_pub + 0.25 * loss_contrast

            adv_loss_val = 0.0
            fm_loss_val = 0.0
            if in_stage_2:
                _, y_d_gs_g, fmap_rs, fmap_gs = discriminator(tactile, pred_tactile)
                adv_loss, fm_loss = generator_adv_and_fm_loss(y_d_gs_g, fmap_rs, fmap_gs)
                loss_g = loss_g + 0.15 * adv_loss + 1.5 * fm_loss
                adv_loss_val = adv_loss.item()
                fm_loss_val = fm_loss.item()

            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer_g.step()

            # Update EMA shadow weights
            ema.update(decoder)

            train_loss_g_total += loss_g.item()

            postfix_data = {
                "G_loss": f"{loss_g.item():.3f}",
                "si_l1": f"{pub_metrics['si_l1']:.3f}",
                "env": f"{pub_metrics['env_loss']:.3f}",
                "edc": f"{pub_metrics['edc_loss']:.3f}",
            }
            if in_stage_2:
                postfix_data["D_loss"] = f"{loss_d.item():.3f}"
                postfix_data["adv"] = f"{adv_loss_val:.3f}"
            train_bar.set_postfix(postfix_data)

        scheduler_g.step()
        if in_stage_2:
            scheduler_d.step()

        avg_train_loss_g = train_loss_g_total / len(train_loader)
        avg_train_loss_d = (train_loss_d_total / len(train_loader)) if in_stage_2 else 0.0

        # ─── 4. Validation Loop with EMA & Publication Composite Score ───
        # Create temporary evaluation model with EMA weights for smoothed, publication-grade output
        eval_decoder = copy.deepcopy(decoder)
        ema.copy_to(eval_decoder)
        eval_decoder.eval()

        val_pub_loss = 0.0
        val_metrics_list = []

        with torch.no_grad():
            for audio, tactile, aux in val_loader:
                audio = audio.to(device)
                tactile = tactile.to(device)

                latent = encoder(audio)
                pred_tactile, dec_aux = eval_decoder(latent, return_aux=True)

                loss_pub, _ = pub_criterion(pred_tactile, tactile, aux=dec_aux)
                val_pub_loss += loss_pub.item()

                p_np = pred_tactile.squeeze(1).cpu().numpy()
                t_np = tactile.squeeze(1).cpu().numpy()
                for b in range(p_np.shape[0]):
                    m = compute_single_pair_metrics(p_np[b], t_np[b], fs=WORKING_SR, max_align_ms=1.5)
                    val_metrics_list.append(m)

        avg_val_loss = val_pub_loss / len(val_loader)
        mean_pcc_al = float(np.mean([m["pearson_r_aligned"] for m in val_metrics_list]))
        mean_pcc_raw = float(np.mean([m["pearson_r_raw"] for m in val_metrics_list]))
        mean_env_r = float(np.mean([m["envelope_pearson_r"] for m in val_metrics_list]))
        mean_sisdr_al = float(np.mean([m["si_sdr_db_aligned"] for m in val_metrics_list]))
        mean_lsd = float(np.mean([m["log_spectral_distance_db"] for m in val_metrics_list]))
        mean_pc_lsd = float(np.mean([m["pacinian_pc_band_lsd_db"] for m in val_metrics_list]))
        mean_edc_rmse = float(np.mean([m["schroeder_edc_rmse_db"] for m in val_metrics_list]))
        mean_peak_err = float(np.mean([m["peak_amplitude_rel_err"] for m in val_metrics_list]))

        # Publication Composite Validation Score:
        # S_val = Pearson_r_al + Env_r + 0.05 * SI-SDR_al - 0.5 * LSD - 0.5 * PC_LSD - 0.1 * EDC_RMSE
        composite_score = (
            mean_pcc_al
            + mean_env_r
            + 0.05 * mean_sisdr_al
            - 0.5 * mean_lsd
            - 0.5 * mean_pc_lsd
            - 0.1 * mean_edc_rmse
        )

        print(
            f"Epoch {epoch:03d} Summary [{stage_name}] -> "
            f"Train G: {avg_train_loss_g:.3f} | Val Loss: {avg_val_loss:.3f} | "
            f"Score: {composite_score:+.3f}\n"
            f"    [PCC(al): {mean_pcc_al:.3f} (raw: {mean_pcc_raw:.3f}) | "
            f"Env-r: {mean_env_r:.3f} | SI-SDR: {mean_sisdr_al:.1f}dB | "
            f"LSD: {mean_lsd:.2f}dB | PC-LSD: {mean_pc_lsd:.2f}dB | EDC-RMSE: {mean_edc_rmse:.2f}dB | "
            f"Peak-Err: {mean_peak_err:.2f}]"
        )

        # ─── 5. Checkpointing on Best Publication Composite Score ───
        if composite_score > best_val_score:
            improvement = composite_score - best_val_score
            best_val_score = composite_score
            patience_counter = 0

            save_payload = {
                "epoch": epoch,
                "model_type": "TransientSEANetDecoder",
                "model_state_dict": eval_decoder.state_dict(),
                "raw_model_state_dict": decoder.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "composite_score": best_val_score,
                "val_loss": avg_val_loss,
                "working_sr": WORKING_SR,
                "window_size": WINDOW_SIZE,
                "metrics": {
                    "pearson_r_aligned": mean_pcc_al,
                    "pearson_r_raw": mean_pcc_raw,
                    "envelope_pearson_r": mean_env_r,
                    "si_sdr_db_aligned": mean_sisdr_al,
                    "lsd_db": mean_lsd,
                    "pc_lsd_db": mean_pc_lsd,
                    "edc_rmse_db": mean_edc_rmse,
                },
            }

            v2_path = checkpoint_dir / "best_sean_decoder_v2.pt"
            torch.save(save_payload, v2_path)

            # Also update best_sean_decoder.pt for drop-in compatibility
            legacy_path = checkpoint_dir / "best_sean_decoder.pt"
            torch.save(save_payload, legacy_path)

            print(f"[*] New best model saved to {v2_path.name} (Score: {best_val_score:+.4f}, delta: +{improvement:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[*] Early stopping triggered after {PATIENCE} epochs without score improvement.")
                break

    print(f"\n[*] Training complete. Best Publication Composite Score: {best_val_score:+.4f}")


if __name__ == "__main__":
    train()

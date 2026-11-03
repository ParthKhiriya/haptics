import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm


class PeriodSubDiscriminator(nn.Module):
    """
    Sub-discriminator operating on 1D waveform reshaped to 2D at a specific prime period.
    Captures periodic resonant frequencies and high-frequency vibrotactile modal ringing.
    """
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 64, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(64, 128, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 128, (5, 1), (1, 1), padding=(2, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(128, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        B, C, T = x.shape
        if T % self.period != 0:
            n_pad = self.period - (T % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            T = T + n_pad
        x = x.view(B, C, T // self.period, self.period)

        fmaps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmaps.append(x)
        x = self.conv_post(x)
        fmaps.append(x)
        return torch.flatten(x, 1, -1), fmaps


class MultiPeriodWaveformDiscriminator(nn.Module):
    """
    Multi-Period Waveform Discriminator (MPD) for vibrotactile impact synthesis.
    Evaluates signal across prime periods p in {2, 3, 5, 7, 11}.
    """
    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodSubDiscriminator(p) for p in periods])

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def discriminator_lsgan_loss(
    disc_real_outputs: list[torch.Tensor], disc_generated_outputs: list[torch.Tensor]
) -> torch.Tensor:
    """Least-squares GAN (LSGAN) loss for discriminator."""
    loss = torch.tensor(0.0, device=disc_real_outputs[0].device)
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1.0 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss = loss + r_loss + g_loss
    return loss


def generator_adv_and_fm_loss(
    disc_outputs: list[torch.Tensor],
    fmap_r: list[list[torch.Tensor]],
    fmap_g: list[list[torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Adversarial generator loss + feature matching loss across all discriminator layers.
    """
    adv_loss = torch.tensor(0.0, device=disc_outputs[0].device)
    for dg in disc_outputs:
        adv_loss = adv_loss + torch.mean((1.0 - dg) ** 2)

    fm_loss = torch.tensor(0.0, device=disc_outputs[0].device)
    for dr_list, dg_list in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr_list, dg_list):
            fm_loss = fm_loss + F.l1_loss(gl, rl.detach())
    return adv_loss, 2.0 * fm_loss


class MaterialContrastiveSignatureLoss(nn.Module):
    """
    Computes paired InfoNCE contrastive loss between the spectral-decay signature
    of the predicted tap waveform and the ground-truth tap waveform.
    Extracts log-magnitude spectrum and Schroeder energy decay curve (EDC).
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def extract_signature(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.squeeze(1)
        # 1. Log-magnitude spectrum pooled to 64 bins
        spec = torch.log(torch.abs(torch.fft.rfft(x, dim=-1)) + 1e-5)
        spec_pool = F.adaptive_avg_pool1d(spec.unsqueeze(1), 64).squeeze(1)

        # 2. Cumulative energy decay curve (Schroeder EDC) pooled to 32 bins
        energy = x ** 2 + 1e-6
        edc = torch.log(
            torch.flip(torch.cumsum(torch.flip(energy, dims=[-1]), dim=-1), dims=[-1]) + 1e-5
        )
        edc_pool = F.adaptive_avg_pool1d(edc.unsqueeze(1), 32).squeeze(1)

        feat = torch.cat([spec_pool, edc_pool], dim=-1)
        return F.normalize(feat, p=2, dim=-1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[0] < 2:
            return torch.tensor(0.0, device=pred.device)
        z_p = self.extract_signature(pred)
        z_t = self.extract_signature(target).detach()

        logits = torch.matmul(z_p, z_t.T) / self.temperature
        labels = torch.arange(pred.shape[0], device=pred.device)
        return F.cross_entropy(logits, labels)

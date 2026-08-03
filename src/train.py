"""Training utilities for keystroke autoencoder models."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.dataset import KeystrokeWindowDataset


def build_model(
    name: str,
    seq_len: int | None = None,
    fusion_type: str | None = None,
    decoder_type: str = "lstm",
) -> nn.Module:
    """Instantiate one of the three thesis architectures.

    `fusion_type`/`decoder_type` only apply to `lstformer`
    (see src/models/lstformer_ae.py -- used by ablate_lstformer.py).
    """
    seq_len = seq_len or config.WINDOW_SIZE

    if name == "lstm":
        from src.models.lstm_ae import LSTMAutoencoder

        return LSTMAutoencoder(
            seq_len=seq_len,
            hidden_dim=config.HIDDEN_DIM,
            latent_dim=config.LATENT_DIM,
            dropout=config.DROPOUT,
        )
    if name == "transformer":
        from src.models.transformer_ae import TransformerAutoencoder

        return TransformerAutoencoder(
            seq_len=seq_len,
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            num_layers=config.NUM_TRANSFORMER_LAYERS,
            dropout=config.DROPOUT,
        )
    if name == "lstformer":
        from src.models.lstformer_ae import LSTFormerAutoencoder

        return LSTFormerAutoencoder(
            seq_len=seq_len,
            hidden_dim=config.HIDDEN_DIM,
            latent_dim=config.LATENT_DIM,
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            num_transformer_layers=config.NUM_TRANSFORMER_LAYERS,
            dropout=config.DROPOUT,
            fusion_type=fusion_type or config.LSTFORMER_FUSION_TYPE,
            decoder_type=decoder_type,
        )
    raise ValueError(f"Unknown model: {name}. Choose lstm, transformer, or lstformer.")


def train_epoch(model, loader, optimizer, criterion, device, scheduler=None, grad_clip_norm=None) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()  # per-step (not per-epoch) LR warmup
        total_loss += loss.item() * batch.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        recon = model(batch)
        loss = criterion(recon, batch)
        total_loss += loss.item() * batch.size(0)
    return total_loss / len(loader.dataset)


def train_model(
    model: nn.Module,
    train_windows,
    val_windows,
    save_path: Path | str | None = None,
    device: str | torch.device | None = None,
    model_name: str | None = None,
) -> dict:
    """
    Train an autoencoder with Adam, MSE loss, and early stopping
    (Section 3.5.2–3.5.3).

    `model_name` selects architecture-specific tuning (Section 3.5.2
    revision): attention-based models ("transformer", "lstformer") get a
    linear LR warm-up over their first `config.WARMUP_STEPS` steps; "lstm"
    gets gradient-norm clipping instead. Pass `model_name=None` (default) to
    fall back to the original flat-LR, no-clipping behavior.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Section 3.9.2: peak VRAM is logged during the (GPU) training phase.
    if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    train_loader = DataLoader(
        KeystrokeWindowDataset(train_windows),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        KeystrokeWindowDataset(val_windows),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()

    scheduler = None
    grad_clip_norm = None
    if config.USE_LR_WARMUP and model_name in ("transformer", "lstformer"):
        warmup_steps = config.WARMUP_STEPS
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps)
        )
    elif model_name == "lstm":
        grad_clip_norm = config.LSTM_GRAD_CLIP_NORM

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "epoch_time": []}

    for epoch in range(1, config.MAX_EPOCHS + 1):
        t0 = time.perf_counter()
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler=scheduler, grad_clip_norm=grad_clip_norm,
        )
        val_loss = eval_epoch(model, val_loader, criterion, device)
        epoch_time = time.perf_counter() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_time"].append(epoch_time)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d} | train MSE={train_loss:.6f} | "
                f"val MSE={val_loss:.6f} | {epoch_time:.2f}s"
            )

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": best_state, "val_loss": best_val_loss}, save_path)

    history["best_val_loss"] = best_val_loss
    if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
        history["peak_vram_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        history["peak_vram_mb"] = None
    return history


@torch.no_grad()
def reconstruction_errors(model, windows, device=None) -> torch.Tensor:
    """Per-window MSE reconstruction error (anomaly score)."""
    device = device or next(model.parameters()).device
    model.eval()
    loader = DataLoader(KeystrokeWindowDataset(windows), batch_size=config.BATCH_SIZE)
    errors = []
    for batch in loader:
        batch = batch.to(device)
        recon = model(batch)
        mse = ((recon - batch) ** 2).mean(dim=(1, 2))
        errors.append(mse.cpu())
    return torch.cat(errors)

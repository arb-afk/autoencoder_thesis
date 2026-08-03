"""Training utilities for keystroke autoencoder models (TensorFlow/Keras)."""

from __future__ import annotations

import time
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

import config


def build_model(name: str, seq_len: int | None = None) -> keras.Model:
    """Instantiate one of the three thesis architectures."""
    seq_len = seq_len or config.WINDOW_SIZE

    if name == "lstm":
        from src.models.lstm_ae import build_lstm_autoencoder

        return build_lstm_autoencoder(
            seq_len=seq_len,
            feature_dim=config.FEATURE_DIM,
            hidden_dim=config.HIDDEN_DIM,
            latent_dim=config.LATENT_DIM,
            dropout=config.DROPOUT,
        )
    if name == "transformer":
        from src.models.transformer_ae import build_transformer_autoencoder

        return build_transformer_autoencoder(
            seq_len=seq_len,
            feature_dim=config.FEATURE_DIM,
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            num_layers=config.NUM_TRANSFORMER_LAYERS,
            dropout=config.DROPOUT,
        )
    if name == "lstformer":
        from src.models.lstformer_ae import build_lstformer_autoencoder

        return build_lstformer_autoencoder(
            seq_len=seq_len,
            feature_dim=config.FEATURE_DIM,
            hidden_dim=config.HIDDEN_DIM,
            latent_dim=config.LATENT_DIM,
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            num_transformer_layers=config.NUM_TRANSFORMER_LAYERS,
            dropout=config.DROPOUT,
            fusion_type=config.LSTFORMER_FUSION_TYPE,
        )
    raise ValueError(f"Unknown model: {name}. Choose lstm, transformer, or lstformer.")


class _EpochTimer(keras.callbacks.Callback):
    """Records wall-clock seconds per epoch (mirrors the PyTorch version's history)."""

    def __init__(self):
        super().__init__()
        self.times: list[float] = []
        self._t0 = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.perf_counter() - self._t0
        self.times.append(elapsed)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:3d} | train MSE={logs.get('loss'):.6f} | "
                f"val MSE={logs.get('val_loss'):.6f} | {elapsed:.2f}s"
            )


def _device_scope(device: str):
    if device in ("cuda", "gpu", "/GPU:0"):
        return tf.device("/GPU:0")
    return tf.device("/CPU:0")


class _LinearWarmup(keras.optimizers.schedules.LearningRateSchedule):
    """Linearly ramps LR from 0 to `base_lr` over `warmup_steps` steps, then holds constant.

    Mirrors the PyTorch version's LambdaLR warmup (see src/train.py) -- Section
    3.5.2 revision: attention-based models get a step-based LR warm-up instead
    of the original flat LR shared by all three architectures.
    """

    def __init__(self, base_lr: float, warmup_steps: int):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        return self.base_lr * tf.minimum(1.0, (step + 1.0) / warmup_steps)

    def get_config(self):
        return {"base_lr": self.base_lr, "warmup_steps": self.warmup_steps}


def _build_optimizer(model_name: str | None) -> keras.optimizers.Optimizer:
    """Architecture-specific tuning (Section 3.5.2 revision, see top-level
    README Section 7.6): Transformer/LST-Former get an LR warm-up (attention
    layers are sensitive to large early gradients); LSTM gets gradient-norm
    clipping instead (vanilla LSTMs are more prone to occasional gradient
    spikes). `model_name=None` falls back to the original flat-LR behavior.
    """
    if config.USE_LR_WARMUP and model_name in ("transformer", "lstformer"):
        schedule = _LinearWarmup(config.LEARNING_RATE, config.WARMUP_STEPS)
        return keras.optimizers.Adam(learning_rate=schedule)
    if model_name == "lstm":
        return keras.optimizers.Adam(learning_rate=config.LEARNING_RATE, clipnorm=config.LSTM_GRAD_CLIP_NORM)
    return keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)


def train_model(
    model: keras.Model,
    train_windows: np.ndarray,
    val_windows: np.ndarray,
    save_path: Path | str | None = None,
    device: str | None = None,
    model_name: str | None = None,
) -> dict:
    """
    Train an autoencoder with Adam, MSE loss, and early stopping
    (Section 3.5.2-3.5.3). Also logs peak VRAM (Section 3.9.2) when a GPU
    is used -- see the CPU/GPU note in the top-level README for why native
    TensorFlow-on-Windows generally has no GPU available.

    `model_name` selects architecture-specific tuning -- see `_build_optimizer`.
    """
    device = device or ("cuda" if tf.config.list_physical_devices("GPU") else "cpu")
    using_gpu = device in ("cuda", "gpu", "/GPU:0") and bool(tf.config.list_physical_devices("GPU"))

    if using_gpu:
        try:
            tf.config.experimental.reset_memory_stats("GPU:0")
        except (ValueError, RuntimeError):
            pass

    with _device_scope(device):
        model.compile(
            optimizer=_build_optimizer(model_name),
            loss=keras.losses.MeanSquaredError(),
        )

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        )
        timer = _EpochTimer()

        history = model.fit(
            train_windows,
            train_windows,
            validation_data=(val_windows, val_windows),
            epochs=config.MAX_EPOCHS,
            batch_size=config.BATCH_SIZE,
            callbacks=[early_stop, timer],
            verbose=0,
        )

    stopped_epoch = len(history.history["loss"])
    if stopped_epoch < config.MAX_EPOCHS:
        print(f"  Early stopping at epoch {stopped_epoch}")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path = save_path.with_suffix(".weights.h5")
        model.save_weights(str(weights_path))

    peak_vram_mb = None
    if using_gpu:
        try:
            mem_info = tf.config.experimental.get_memory_info("GPU:0")
            peak_vram_mb = mem_info["peak"] / (1024 * 1024)
        except (ValueError, RuntimeError):
            peak_vram_mb = None

    return {
        "train_loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "epoch_time": timer.times,
        "best_val_loss": float(min(history.history["val_loss"])),
        "peak_vram_mb": peak_vram_mb,
    }


def reconstruction_errors(model: keras.Model, windows: np.ndarray, device: str | None = None) -> np.ndarray:
    """Per-window MSE reconstruction error (anomaly score)."""
    if len(windows) == 0:
        return np.empty((0,), dtype=np.float32)
    device = device or "cpu"
    with _device_scope(device):
        recon = model.predict(windows, batch_size=config.BATCH_SIZE, verbose=0)
    return np.mean((recon - windows) ** 2, axis=(1, 2))

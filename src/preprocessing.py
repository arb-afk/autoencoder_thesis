"""Keystroke feature extraction and sequence generation for Clarkson II dataset.

Leak-free protocol (thesis §3.3.2–3.5.1):
  1. Extract HT/FT in chronological order.
  2. Split the feature sequence into train / validation / test.
  3. Fit per-user mean/std on the training portion only; apply to val/test.
  4. Build sliding windows independently inside each portion (no cross-boundary windows).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import NamedTuple

import numpy as np

from config import (
    FLIGHT_TIME_CLIP_MS,
    RANDOM_SEED,
    STRIDE,
    TRAIN_RATIO,
    VAL_RATIO,
    WINDOW_SIZE,
)

# Clarkson II timestamps are .NET DateTime Ticks: 1 tick = 100 ns = 1e-4 ms.
TICKS_TO_MS = 1e-4


class KeystrokeEvent(NamedTuple):
    timestamp: int
    action: int  # 0 = press, 1 = release
    key: str


class UserSplits(NamedTuple):
    """Per-user window tensors after leak-free split → normalize → window."""

    train: np.ndarray  # (n_train, N, 2)
    val: np.ndarray
    test: np.ndarray
    train_mean: np.ndarray
    train_std: np.ndarray
    n_features: int


def load_keystroke_file(path: Path | str) -> list[KeystrokeEvent]:
    """Load a Clarkson II keystroke log (timestamp, action, key)."""
    events: list[KeystrokeEvent] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            events.append(KeystrokeEvent(int(parts[0]), int(parts[1]), parts[2]))
    return events


def extract_ht_ft(events: list[KeystrokeEvent]) -> np.ndarray:
    """
    Extract Hold Time (HT) and Flight Time (FT) features per keystroke.

    HT_i = t_release,i - t_press,i
    FT_i = t_press,i+1 - t_release,i
    """
    presses: dict[str, int] = {}
    pairs: list[tuple[int, int]] = []  # (press_ts, release_ts) in order

    for ev in events:
        if ev.action == 0:
            presses[ev.key] = ev.timestamp
        elif ev.action == 1 and ev.key in presses:
            pairs.append((presses[ev.key], ev.timestamp))
            del presses[ev.key]

    if len(pairs) < 2:
        return np.empty((0, 2), dtype=np.float32)

    features = np.zeros((len(pairs), 2), dtype=np.float32)
    for i, (press_ts, release_ts) in enumerate(pairs):
        features[i, 0] = (release_ts - press_ts) * TICKS_TO_MS  # HT in ms

    for i in range(len(pairs) - 1):
        ft_ms = (pairs[i + 1][0] - pairs[i][1]) * TICKS_TO_MS
        features[i, 1] = min(ft_ms, FLIGHT_TIME_CLIP_MS)  # clip FT (Section 3.3.2)

    # Last keystroke has no successor flight time; use previous FT as fallback
    if len(pairs) > 1:
        features[-1, 1] = features[-2, 1]

    return features


def split_feature_sequence(
    features: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological 70/10/20 split on the keystroke feature sequence."""
    n = len(features)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return features[:train_end], features[train_end:val_end], features[val_end:]


def zscore_fit(train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-feature mean/std on the training portion only."""
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def zscore_apply(
    features: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Apply training-set z-score statistics to a feature matrix."""
    if len(features) == 0:
        return features.astype(np.float32)
    return ((features - mean) / std).astype(np.float32)


def zscore_normalize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit+apply z-score on one matrix (legacy helper; prefer train-only fit)."""
    mean, std = zscore_fit(features)
    return zscore_apply(features, mean, std), mean, std


def sliding_windows(
    features: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> np.ndarray:
    """Create overlapping sliding windows of shape (num_windows, window_size, 2)."""
    if len(features) < window_size:
        return np.empty((0, window_size, features.shape[1]), dtype=np.float32)

    windows = []
    for start in range(0, len(features) - window_size + 1, stride):
        windows.append(features[start : start + window_size])
    return np.stack(windows, axis=0)


def load_user_splits(
    path: Path | str,
    window_size: int = WINDOW_SIZE,
    stride: int | None = None,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> UserSplits | None:
    """Load one user with leak-free split → train-only normalize → per-split windows.

    Returns None when the feature sequence is empty.
    """
    if stride is None:
        stride = max(1, window_size // 2)

    events = load_keystroke_file(path)
    features = extract_ht_ft(events)
    if len(features) == 0:
        return None

    train_f, val_f, test_f = split_feature_sequence(
        features, train_ratio=train_ratio, val_ratio=val_ratio
    )
    if len(train_f) == 0:
        empty = np.empty((0, window_size, 2), dtype=np.float32)
        mean = np.zeros(2, dtype=np.float32)
        std = np.ones(2, dtype=np.float32)
        return UserSplits(empty, empty, empty, mean, std, len(features))

    mean, std = zscore_fit(train_f)
    train_w = sliding_windows(
        zscore_apply(train_f, mean, std), window_size=window_size, stride=stride
    )
    val_w = sliding_windows(
        zscore_apply(val_f, mean, std), window_size=window_size, stride=stride
    )
    test_w = sliding_windows(
        zscore_apply(test_f, mean, std), window_size=window_size, stride=stride
    )
    return UserSplits(train_w, val_w, test_w, mean, std, len(features))


def load_all_users(
    data_dir: Path | str,
    window_size: int = WINDOW_SIZE,
    stride: int | None = None,
) -> dict[str, UserSplits]:
    """Load leak-free train/val/test windows for every user file in the dataset."""
    data_dir = Path(data_dir)
    users: dict[str, UserSplits] = {}
    for path in sorted(data_dir.iterdir()):
        if path.is_file():
            splits = load_user_splits(path, window_size=window_size, stride=stride)
            if splits is None:
                continue
            # Keep users that produce at least one window in any partition so
            # callers can decide skip criteria (usually train length).
            if len(splits.train) + len(splits.val) + len(splits.test) > 0:
                users[path.name] = splits
    return users


def chronological_split(
    windows: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy window-level split. Prefer load_user_splits / UserSplits instead.

    Kept for scripts that still pass a single window tensor; does NOT prevent
    boundary-crossing leakage when windows were built on the full sequence.
    """
    n = len(windows)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return windows[:train_end], windows[train_end:val_end], windows[val_end:]


def impostor_rng(enrolled_user_id: str, seed: int = RANDOM_SEED) -> np.random.Generator:
    """Deterministic RNG for impostor sampling, independent of training RNG state."""
    digest = hashlib.sha256(f"{seed}:{enrolled_user_id}".encode("utf-8")).digest()
    seed_int = int.from_bytes(digest[:8], "little") % (2**63 - 1)
    return np.random.default_rng(seed_int)


def sample_zero_effort_impostors(
    enrolled_user_id: str,
    all_users: dict[str, UserSplits],
    n_genuine_test: int,
    seed: int = RANDOM_SEED,
) -> np.ndarray | None:
    """Sample held-out impostor test windows with a fixed per-user RNG.

    Uses the same indices for a given enrolled user regardless of when this is
    called relative to model training, so every architecture sees the same
    impostor set for that user.
    """
    rng = impostor_rng(enrolled_user_id, seed=seed)
    chunks: list[np.ndarray] = []
    for other_id, other in all_users.items():
        if other_id == enrolled_user_id:
            continue
        other_test = other.test
        if len(other_test) == 0:
            continue
        n_sample = min(len(other_test), max(10, n_genuine_test // 5))
        idx = rng.choice(len(other_test), size=n_sample, replace=False)
        chunks.append(other_test[idx])
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)

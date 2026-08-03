"""Keystroke feature extraction and sequence generation for Clarkson II dataset."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

from config import FLIGHT_TIME_CLIP_MS, STRIDE, WINDOW_SIZE

# Clarkson II timestamps are .NET DateTime Ticks: 1 tick = 100 ns = 1e-4 ms.
# Verified empirically: raw tick diffs for real keystrokes decode to ~50-150ms
# hold times only under this conversion, and raw timestamps decode to dates
# in 2015-2018, matching the dataset's known collection period.
TICKS_TO_MS = 1e-4


class KeystrokeEvent(NamedTuple):
    timestamp: int
    action: int  # 0 = press, 1 = release
    key: str


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


def zscore_normalize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-user Z-score normalization (Section 3.3.2)."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (features - mean) / std
    return normalized.astype(np.float32), mean, std


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


def load_user_windows(
    path: Path | str,
    window_size: int = WINDOW_SIZE,
    stride: int | None = None,
) -> np.ndarray:
    """Full pipeline: load file → extract HT/FT → normalize → window.

    `stride` defaults to 50% overlap (`window_size // 2`), matching Section 3.3.3.
    Pass an explicit stride when sweeping window lengths so longer windows do
    not silently inherit the config default of 25.
    """
    if stride is None:
        stride = max(1, window_size // 2)
    events = load_keystroke_file(path)
    features = extract_ht_ft(events)
    if len(features) == 0:
        return np.empty((0, window_size, FEATURE_DIM := 2), dtype=np.float32)
    normalized, _, _ = zscore_normalize(features)
    return sliding_windows(normalized, window_size=window_size, stride=stride)


def chronological_split(
    windows: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological 70/10/20 split (Section 3.5.1)."""
    n = len(windows)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return windows[:train_end], windows[train_end:val_end], windows[val_end:]


def load_all_users(
    data_dir: Path | str,
    window_size: int = WINDOW_SIZE,
    stride: int | None = None,
) -> dict[str, np.ndarray]:
    """Load sliding windows for every user file in the dataset directory."""
    data_dir = Path(data_dir)
    users: dict[str, np.ndarray] = {}
    for path in sorted(data_dir.iterdir()):
        if path.is_file():
            windows = load_user_windows(path, window_size=window_size, stride=stride)
            if len(windows) > 0:
                users[path.name] = windows
    return users

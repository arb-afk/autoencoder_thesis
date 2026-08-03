"""Biometric evaluation metrics (Section 3.8)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import auc, roc_curve


def compute_eer(genuine_scores: np.ndarray, imposter_scores: np.ndarray) -> float:
    """
    Equal Error Rate — threshold where FAR ≈ FRR.

    Lower scores = more anomalous for imposters; higher = genuine.
    We treat reconstruction error as anomaly score (higher = more anomalous).
    """
    y_true = np.concatenate([
        np.zeros(len(genuine_scores)),
        np.ones(len(imposter_scores)),
    ])
    y_score = np.concatenate([genuine_scores, imposter_scores])

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[eer_idx] + fnr[eer_idx]) / 2)


def compute_validation_threshold(val_errors: np.ndarray) -> float:
    """
    Decision threshold (theta) set during the Validation Phase (Section 3.6.2),
    not derived from test-set data. Uses the maximum reconstruction error
    observed on genuine validation windows, so the operating point accepts
    all validation-time genuine behavior before being exposed to test/imposter data.
    """
    return float(np.max(val_errors))


def compute_far_frr(
    genuine_scores: np.ndarray,
    imposter_scores: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    """
    False Acceptance Rate and False Rejection Rate at a given threshold.
    The threshold must come from the validation phase (Section 3.6.2), not
    from the test set itself. Returns (FAR, FRR, threshold).
    """
    far = float(np.mean(imposter_scores < threshold))  # imposter accepted as genuine
    frr = float(np.mean(genuine_scores >= threshold))  # genuine rejected
    return far, frr, threshold


def compute_auc(genuine_scores: np.ndarray, imposter_scores: np.ndarray) -> float:
    """Area Under the ROC Curve."""
    y_true = np.concatenate([
        np.zeros(len(genuine_scores)),
        np.ones(len(imposter_scores)),
    ])
    y_score = np.concatenate([genuine_scores, imposter_scores])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(auc(fpr, tpr))


def evaluate_user(
    genuine_errors: np.ndarray,
    imposter_errors: np.ndarray,
    val_errors: np.ndarray,
) -> dict[str, float]:
    """Full biometric evaluation for one enrolled user."""
    eer = compute_eer(genuine_errors, imposter_errors)
    auc_score = compute_auc(genuine_errors, imposter_errors)
    threshold = compute_validation_threshold(val_errors)
    far, frr, threshold = compute_far_frr(genuine_errors, imposter_errors, threshold)
    return {
        "eer": eer,
        "auc": auc_score,
        "far": far,
        "frr": frr,
        "threshold": threshold,
        "genuine_mean": float(genuine_errors.mean()),
        "imposter_mean": float(imposter_errors.mean()),
    }


def measure_inference_latency(model, sample_input, device="cpu", n_warmup=10, n_runs=100) -> float:
    """
    Average inference latency in milliseconds (Section 3.9.1). Defaults to
    CPU per Section 3.7.1 ("testing phase will prioritize CPU-only inference
    benchmarks"), moving the model off whatever device it was trained on.
    """
    import time

    import torch

    original_device = next(model.parameters()).device
    model = model.to(device).eval()
    sample = sample_input.to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(sample)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(sample)
            times.append((time.perf_counter() - t0) * 1000)
    model.to(original_device)
    return float(np.mean(times))

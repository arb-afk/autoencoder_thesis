"""Biometric evaluation metrics (Section 3.8) -- TensorFlow/Keras port.

The FAR/FRR/EER/AUC math is framework-agnostic (NumPy + scikit-learn), so
it is identical to ../../src/evaluate.py. Only `measure_inference_latency`
differs, since it must call the Keras model instead of a PyTorch module.
"""

from __future__ import annotations

import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, roc_curve


def compute_eer(genuine_scores: np.ndarray, imposter_scores: np.ndarray) -> float:
    """Equal Error Rate -- threshold where FAR ≈ FRR (Equation 9)."""
    y_true = np.concatenate([np.zeros(len(genuine_scores)), np.ones(len(imposter_scores))])
    y_score = np.concatenate([genuine_scores, imposter_scores])

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[eer_idx] + fnr[eer_idx]) / 2)


def compute_validation_threshold(val_errors: np.ndarray) -> float:
    """
    Decision threshold (theta) set during the Validation Phase (Section 3.6.2).
    Uses the maximum reconstruction error observed on genuine validation
    windows, so the operating point accepts all validation-time genuine
    behavior before being exposed to test/imposter data.
    """
    return float(np.max(val_errors))


def compute_far_frr(
    genuine_scores: np.ndarray,
    imposter_scores: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    """FAR (Equation 7) / FRR (Equation 8) at a validation-derived threshold."""
    far = float(np.mean(imposter_scores < threshold))  # imposter accepted as genuine
    frr = float(np.mean(genuine_scores >= threshold))  # genuine rejected
    return far, frr, threshold


def compute_auc(genuine_scores: np.ndarray, imposter_scores: np.ndarray) -> float:
    """Area Under the ROC Curve."""
    y_true = np.concatenate([np.zeros(len(genuine_scores)), np.ones(len(imposter_scores))])
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


def measure_inference_latency(
    model,
    sample_input: np.ndarray,
    device: str = "cpu",
    n_warmup: int = 10,
    n_runs: int = 100,
) -> float:
    """
    Average inference latency in milliseconds (Section 3.9.1). Defaults to
    CPU per Section 3.7.1 ("testing phase will prioritize CPU-only inference
    benchmarks"), regardless of which device was used for training.

    The forward pass is wrapped in tf.function so it runs as a compiled
    graph rather than raw Python/eager mode -- calling a Keras model
    directly in eager mode has large per-call Python overhead (tracing,
    dispatch) that would make TensorFlow look artificially ~100x slower
    than the PyTorch version for the exact same architecture. Compiling
    once (during warmup) and reusing the graph gives a fair, representative
    number, consistent with how these models would actually be deployed.
    """
    tf_device = "/GPU:0" if device in ("cuda", "gpu", "/GPU:0") else "/CPU:0"
    sample = tf.convert_to_tensor(sample_input, dtype=tf.float32)

    @tf.function
    def _forward(x):
        return model(x, training=False)

    with tf.device(tf_device):
        for _ in range(n_warmup):
            _forward(sample)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _forward(sample)
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))

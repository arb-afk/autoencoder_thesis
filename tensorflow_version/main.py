"""
Main entry point (TensorFlow/Keras version): train and evaluate LSTM,
Transformer, and LST-Former autoencoders on the Clarkson II keystroke
dataset. Mirrors ../main.py (PyTorch) so results are comparable.

Usage:
    python main.py --model lstm
    python main.py --model all
    python main.py --model lstformer --user 118630
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.evaluate import evaluate_user, measure_inference_latency
from src.preprocessing import chronological_split, load_all_users
from src.train import build_model, reconstruction_errors, train_model


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Reproducibility -- see README Section 7.6 / root main.py's set_seed."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def run_single_user(model_name: str, user_id: str, all_users: dict, device: str) -> dict:
    """Train on one user, test with zero-effort imposters from other users."""
    windows = all_users[user_id]
    train, val, test = chronological_split(windows)

    if len(train) < config.BATCH_SIZE:
        return {"user": user_id, "status": "skipped", "reason": "insufficient training data"}

    seq_len = train.shape[1]
    model = build_model(model_name, seq_len=seq_len)

    save_path = config.MODEL_DIR / model_name / user_id
    history = train_model(model, train, val, save_path=save_path, device=device, model_name=model_name)

    genuine_errors = reconstruction_errors(model, test, device)
    val_errors = reconstruction_errors(model, val, device)

    imposter_windows = []
    for other_id, other_w in all_users.items():
        if other_id == user_id:
            continue
        _, _, other_test = chronological_split(other_w)
        if len(other_test) > 0:
            n_sample = min(len(other_test), max(10, len(test) // 5))
            idx = np.random.choice(len(other_test), n_sample, replace=False)
            imposter_windows.append(other_test[idx])

    if not imposter_windows:
        return {"user": user_id, "status": "skipped", "reason": "no imposter data"}

    imposter_all = np.concatenate(imposter_windows, axis=0)
    imposter_errors = reconstruction_errors(model, imposter_all, device)

    metrics = evaluate_user(genuine_errors, imposter_errors, val_errors)

    # Section 3.7.1: testing phase prioritizes CPU-only inference benchmarks.
    sample = np.random.randn(1, seq_len, config.FEATURE_DIM).astype(np.float32)
    latency_ms = measure_inference_latency(model, sample, device="cpu")

    return {
        "user": user_id,
        "status": "ok",
        "epochs_trained": len(history["train_loss"]),
        "best_val_loss": history["best_val_loss"],
        "num_params": model.count_params(),
        "inference_latency_ms": latency_ms,
        "peak_vram_mb": history.get("peak_vram_mb"),
        "train_device": device,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Keystroke anomaly detection autoencoders (TensorFlow)")
    parser.add_argument(
        "--model",
        choices=["lstm", "transformer", "lstformer", "all"],
        default="all",
        help="Which architecture to train",
    )
    parser.add_argument("--user", type=str, default=None, help="Single user ID to process")
    parser.add_argument("--max-users", type=int, default=5, help="Max users when running all")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save results.json (default: outputs/results.json). Use a "
        "distinct path (e.g. outputs/results_gpu.json) to keep a GPU run separate "
        "from an existing CPU run instead of overwriting it.",
    )
    args = parser.parse_args()

    gpu_available = bool(tf.config.list_physical_devices("GPU"))
    device = args.device or ("cuda" if gpu_available else "cpu")
    print(f"Device: {device} (GPU available: {gpu_available})")
    print(f"Loading Clarkson II dataset from {config.DATA_DIR} ...")

    all_users = load_all_users(config.DATA_DIR)
    print(f"Loaded {len(all_users)} users")

    user_ids = [args.user] if args.user else list(all_users.keys())[: args.max_users]
    models = ["lstm", "transformer", "lstformer"] if args.model == "all" else [args.model]

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output or (config.OUTPUT_DIR / "results.json")
    all_results = {}
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)  # merge so re-running one model doesn't drop the others

    for model_name in models:
        set_seed()  # each architecture starts from the same RNG state -- fair, reproducible
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name.upper()}")
        print(f"{'=' * 60}")
        results = []
        for uid in user_ids:
            if uid not in all_users:
                print(f"  User {uid} not found, skipping")
                continue
            print(f"\n  User {uid}:")
            result = run_single_user(model_name, uid, all_users, device)
            results.append(result)
            if result["status"] == "ok":
                vram_str = (
                    f" | VRAM={result['peak_vram_mb']:.1f}MB"
                    if result.get("peak_vram_mb") is not None
                    else ""
                )
                print(
                    f"    EER={result['eer']:.4f} | AUC={result['auc']:.4f} | "
                    f"Latency={result['inference_latency_ms']:.2f}ms | "
                    f"Params={result['num_params']:,}{vram_str}"
                )
            else:
                print(f"    Skipped: {result.get('reason', 'unknown')}")

        ok_results = [r for r in results if r["status"] == "ok"]
        if ok_results:
            summary = {
                "model": model_name,
                "users_evaluated": len(ok_results),
                "mean_eer": float(np.mean([r["eer"] for r in ok_results])),
                "mean_auc": float(np.mean([r["auc"] for r in ok_results])),
                "mean_latency_ms": float(np.mean([r["inference_latency_ms"] for r in ok_results])),
                "mean_params": float(np.mean([r["num_params"] for r in ok_results])),
            }
            print(
                f"\n  Summary ({model_name}): EER={summary['mean_eer']:.4f}, "
                f"AUC={summary['mean_auc']:.4f}, Latency={summary['mean_latency_ms']:.2f}ms"
            )
        else:
            summary = {"model": model_name, "users_evaluated": 0}

        all_results[model_name] = {"summary": summary, "per_user": results}

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

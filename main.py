"""
Main entry point: train and evaluate LSTM, Transformer, and LST-Former
autoencoders on the Clarkson II keystroke dataset.

Usage:
    python main.py --model lstm
    python main.py --model all
    python main.py --model lstformer --user 118630
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.evaluate import evaluate_user, measure_inference_latency
from src.preprocessing import load_all_users, sample_zero_effort_impostors
from src.train import build_model, reconstruction_errors, train_model


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Reproducibility (README Section 7.6): without this, re-running with
    identical config gives different EER/AUC each time from random weight
    init and data shuffling alone. Impostor indices use a separate per-user RNG."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_user(model_name: str, user_id: str, all_users: dict, device: str) -> dict:
    """Train on one user, test with zero-effort imposters from other users."""
    splits = all_users[user_id]
    train, val, test = splits.train, splits.val, splits.test

    if len(train) < config.BATCH_SIZE:
        return {"user": user_id, "status": "skipped", "reason": "insufficient training data"}

    seq_len = train.shape[1]
    model = build_model(model_name, seq_len=seq_len)

    save_path = config.MODEL_DIR / model_name / f"{user_id}.pt"
    history = train_model(model, train, val, save_path=save_path, device=device, model_name=model_name)

    # Genuine test scores + validation scores (validation sets the decision threshold, Section 3.6.2)
    genuine_errors = reconstruction_errors(model, test, device).numpy()
    val_errors = reconstruction_errors(model, val, device).numpy()

    # Fixed per-user impostor sample (independent of training RNG; same across architectures)
    imposter_all = sample_zero_effort_impostors(user_id, all_users, n_genuine_test=len(test))
    if imposter_all is None:
        return {"user": user_id, "status": "skipped", "reason": "no imposter data"}

    imposter_errors = reconstruction_errors(model, imposter_all, device).numpy()

    metrics = evaluate_user(genuine_errors, imposter_errors, val_errors)

    # Section 3.7.1: testing phase prioritizes CPU-only inference benchmarks,
    # regardless of which device was used for training/enrollment.
    sample = torch.randn(1, seq_len, config.FEATURE_DIM)
    latency_ms = measure_inference_latency(model, sample, device="cpu")

    return {
        "user": user_id,
        "status": "ok",
        "epochs_trained": len(history["train_loss"]),
        "best_val_loss": history["best_val_loss"],
        "num_params": sum(p.numel() for p in model.parameters()),
        "inference_latency_ms": latency_ms,
        "peak_vram_mb": history.get("peak_vram_mb"),
        "train_device": device,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Keystroke anomaly detection autoencoders")
    parser.add_argument(
        "--model",
        choices=["lstm", "transformer", "lstformer", "all"],
        default="all",
        help="Which architecture to train",
    )
    parser.add_argument("--user", type=str, default=None, help="Single user ID to process")
    parser.add_argument("--max-users", type=int, default=5, help="Max users when running all")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading Clarkson II dataset from {config.DATA_DIR} ...")

    all_users = load_all_users(config.DATA_DIR)
    print(f"Loaded {len(all_users)} users")

    user_ids = [args.user] if args.user else list(all_users.keys())[: args.max_users]
    models = ["lstm", "transformer", "lstformer"] if args.model == "all" else [args.model]

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.OUTPUT_DIR / "results.json"
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
            print(f"\n  Summary ({model_name}): EER={summary['mean_eer']:.4f}, "
                  f"AUC={summary['mean_auc']:.4f}, Latency={summary['mean_latency_ms']:.2f}ms")
        else:
            summary = {"model": model_name, "users_evaluated": 0}

        all_results[model_name] = {"summary": summary, "per_user": results}

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

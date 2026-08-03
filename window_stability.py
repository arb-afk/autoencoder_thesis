"""
Window-length × architecture stability sweep (thesis Section 3.x / RQ on
how input-sequence length affects reconstruction error and classification
stability).

Builds the advisor-recommended matrix:

  architecture × window length × {EER, FAR, FRR, AUC,
                                  genuine/imposter recon-error stats,
                                  latency, peak VRAM, params}

Methodology is identical to main.py except that WINDOW_SIZE is swept
(default: config.SEQUENCE_LENGTHS = [25, 50, 100]) with 50% overlap stride
at each length. Models are saved under outputs/models_window_stability/
so they do not overwrite the N=50 main-run checkpoints.

Usage:
    python window_stability.py --max-users 96 --device cuda
    python window_stability.py --lengths 25 50 100 --max-users 30 --device cuda
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
from src.preprocessing import chronological_split, load_all_users
from src.train import build_model, reconstruction_errors, train_model


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_user(
    model_name: str,
    user_id: str,
    all_users: dict,
    device: str,
    seq_len: int,
    save_root: Path,
) -> dict:
    windows = all_users[user_id]
    train, val, test = chronological_split(windows)

    if len(train) < config.BATCH_SIZE:
        return {"user": user_id, "status": "skipped", "reason": "insufficient training data"}

    model = build_model(model_name, seq_len=seq_len)
    save_path = save_root / model_name / f"N{seq_len}" / f"{user_id}.pt"
    history = train_model(
        model, train, val, save_path=save_path, device=device, model_name=model_name
    )

    genuine_errors = reconstruction_errors(model, test, device).numpy()
    val_errors = reconstruction_errors(model, val, device).numpy()

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
    imposter_errors = reconstruction_errors(model, imposter_all, device).numpy()
    metrics = evaluate_user(genuine_errors, imposter_errors, val_errors)

    sample = torch.randn(1, seq_len, config.FEATURE_DIM)
    latency_ms = measure_inference_latency(model, sample, device="cpu")

    genuine_std = float(genuine_errors.std()) if len(genuine_errors) else 0.0
    imposter_std = float(imposter_errors.std()) if len(imposter_errors) else 0.0

    return {
        "user": user_id,
        "status": "ok",
        "window_size": seq_len,
        "epochs_trained": len(history["train_loss"]),
        "best_val_loss": history["best_val_loss"],
        "num_params": sum(p.numel() for p in model.parameters()),
        "inference_latency_ms": latency_ms,
        "peak_vram_mb": history.get("peak_vram_mb"),
        "train_device": device,
        "genuine_std": genuine_std,
        "imposter_std": imposter_std,
        "recon_gap": float(metrics["imposter_mean"] - metrics["genuine_mean"]),
        **metrics,
    }


def summarize(model_name: str, seq_len: int, results: list[dict]) -> dict:
    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        return {
            "model": model_name,
            "window_size": seq_len,
            "users_evaluated": 0,
            "users_skipped": len(results),
        }

    def mean(key: str) -> float:
        vals = [r[key] for r in ok if r.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "model": model_name,
        "window_size": seq_len,
        "users_evaluated": len(ok),
        "users_skipped": len(results) - len(ok),
        "mean_eer": mean("eer"),
        "mean_auc": mean("auc"),
        "mean_far": mean("far"),
        "mean_frr": mean("frr"),
        "mean_genuine_recon": mean("genuine_mean"),
        "mean_imposter_recon": mean("imposter_mean"),
        "mean_recon_gap": mean("recon_gap"),
        "mean_latency_ms": mean("inference_latency_ms"),
        "mean_peak_vram_mb": mean("peak_vram_mb"),
        "mean_params": mean("num_params"),
    }


def write_markdown_matrix(all_results: dict, out_path: Path) -> None:
    """Human-readable architecture × window matrix for the thesis Results chapter."""
    models = ["lstm", "transformer", "lstformer"]
    lengths = sorted({int(k.split("_N")[-1]) for k in all_results if "_N" in k})

    lines = [
        "# Architecture × Window-Length Stability Matrix",
        "",
        "Source: `outputs/window_stability.json` (seeded PyTorch run).",
        "Question: how does authentication / reconstruction / efficiency change",
        "as temporal context (sliding-window length N) increases?",
        "",
        "## Mean EER (lower is better)",
        "",
        "| Window N | LSTM | Transformer | LST-Former |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for n in lengths:
        row = [str(n)]
        for m in models:
            s = all_results.get(f"{m}_N{n}", {}).get("summary", {})
            row.append(f"{s['mean_eer']:.4f}" if s.get("mean_eer") is not None else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Mean AUC (higher is better)",
        "",
        "| Window N | LSTM | Transformer | LST-Former |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for n in lengths:
        row = [str(n)]
        for m in models:
            s = all_results.get(f"{m}_N{n}", {}).get("summary", {})
            row.append(f"{s['mean_auc']:.4f}" if s.get("mean_auc") is not None else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Mean FAR / FRR (at validation-derived threshold)",
        "",
        "| Window N | LSTM FAR/FRR | Transformer FAR/FRR | LST-Former FAR/FRR |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for n in lengths:
        row = [str(n)]
        for m in models:
            s = all_results.get(f"{m}_N{n}", {}).get("summary", {})
            if s.get("mean_far") is None:
                row.append("—")
            else:
                row.append(f"{s['mean_far']:.3f} / {s['mean_frr']:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Reconstruction-error gap (imposter mean − genuine mean)",
        "",
        "Larger positive gap ⇒ clearer separation between genuine and impostor",
        "reconstruction distributions.",
        "",
        "| Window N | LSTM | Transformer | LST-Former |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for n in lengths:
        row = [str(n)]
        for m in models:
            s = all_results.get(f"{m}_N{n}", {}).get("summary", {})
            row.append(
                f"{s['mean_recon_gap']:.4f}" if s.get("mean_recon_gap") is not None else "—"
            )
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Efficiency (mean CPU latency ms / mean peak VRAM MB / params)",
        "",
        "| Window N | LSTM | Transformer | LST-Former |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for n in lengths:
        row = [str(n)]
        for m in models:
            s = all_results.get(f"{m}_N{n}", {}).get("summary", {})
            if s.get("mean_latency_ms") is None:
                row.append("—")
            else:
                vram = s.get("mean_peak_vram_mb")
                vram_s = f"{vram:.1f}" if vram == vram else "?"  # NaN check
                row.append(
                    f"{s['mean_latency_ms']:.2f}ms / {vram_s}MB / {int(s['mean_params']):,}"
                )
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Users evaluated per cell",
        "",
        "Longer windows yield fewer sliding windows per user, so some users",
        "fall below `BATCH_SIZE` and are skipped. Prefer paired comparisons",
        "on the intersection of users who succeed at every N.",
        "",
        "| Window N | LSTM | Transformer | LST-Former |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for n in lengths:
        row = [str(n)]
        for m in models:
            s = all_results.get(f"{m}_N{n}", {}).get("summary", {})
            row.append(str(s.get("users_evaluated", 0)))
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## How to read this for the thesis",
        "",
        "Do **not** force “LST-Former is inherently superior.” Ask instead:",
        "",
        "1. Does any architecture’s EER/AUC *trajectory* improve (or degrade)",
        "   as N grows?",
        "2. Does LST-Former only pull ahead (if at all) at longer windows —",
        "   where local+global complementarity is most testable?",
        "3. Where security is tied, does efficiency (latency / memory) still",
        "   differentiate the designs?",
        "",
        "A flat Transformer ≈ LST-Former trajectory across N is itself evidence",
        "that the LSTM branch may be largely redundant given what self-attention",
        "already captures from HT/FT windows.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Architecture × window-length authentication/efficiency matrix"
    )
    parser.add_argument("--max-users", type=int, default=96)
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=None,
        help="Window lengths to sweep (default: config.SEQUENCE_LENGTHS)",
    )
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lstm", "transformer", "lstformer"],
        choices=["lstm", "transformer", "lstformer"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path (default: outputs/window_stability.json)",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    lengths = args.lengths or list(config.SEQUENCE_LENGTHS)
    out_path = args.output or (config.OUTPUT_DIR / "window_stability.json")
    md_path = out_path.with_suffix(".md")
    save_root = config.OUTPUT_DIR / "models_window_stability"

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict = {}
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)

    print(f"Device: {device}")
    print(f"Window lengths: {lengths}")
    print(f"Models: {args.models}")
    print(f"Max users: {args.max_users}")

    for seq_len in lengths:
        print(f"\n{'#' * 60}")
        print(f"# Loading users at window_size={seq_len} (50% overlap stride)")
        print(f"{'#' * 60}")
        all_users = load_all_users(config.DATA_DIR, window_size=seq_len)
        user_ids = list(all_users.keys())[: args.max_users]
        print(f"Loaded {len(all_users)} users with >=1 window; using first {len(user_ids)}")

        for model_name in args.models:
            cell_key = f"{model_name}_N{seq_len}"
            existing = all_results.get(cell_key, {})
            existing_per_user = {
                r["user"]: r
                for r in existing.get("per_user", [])
                if r.get("status") == "ok"
            }
            # Skip fully completed cells (all requested users already evaluated or skipped)
            if existing.get("summary", {}).get("users_evaluated", 0) > 0 and len(
                existing.get("per_user", [])
            ) >= len(user_ids):
                print(f"\nSkipping {cell_key} (already complete in {out_path.name})")
                continue

            set_seed()
            print(f"\n{'=' * 60}")
            print(f"Model: {model_name.upper()} | N={seq_len}")
            print(f"{'=' * 60}")

            results = list(existing.get("per_user", []))
            done_ids = {r["user"] for r in results}
            for uid in user_ids:
                if uid in done_ids:
                    print(f"\n  User {uid}: already done, skipping")
                    continue
                print(f"\n  User {uid}:")
                result = run_single_user(
                    model_name, uid, all_users, device, seq_len, save_root
                )
                results.append(result)
                done_ids.add(uid)
                if result["status"] == "ok":
                    print(
                        f"    EER={result['eer']:.4f} | AUC={result['auc']:.4f} | "
                        f"gap={result['recon_gap']:.4f} | "
                        f"lat={result['inference_latency_ms']:.2f}ms"
                    )
                else:
                    print(f"    Skipped: {result.get('reason', 'unknown')}")

                # Incremental checkpoint after every user (resume-safe)
                summary = summarize(model_name, seq_len, results)
                all_results[cell_key] = {"summary": summary, "per_user": results}
                with open(out_path, "w") as f:
                    json.dump(all_results, f, indent=2)

            summary = summarize(model_name, seq_len, results)
            print(
                f"\n  Summary {cell_key}: n={summary.get('users_evaluated', 0)} | "
                f"EER={summary.get('mean_eer', float('nan')):.4f} | "
                f"AUC={summary.get('mean_auc', float('nan')):.4f}"
            )
            all_results[cell_key] = {"summary": summary, "per_user": results}
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)
            write_markdown_matrix(all_results, md_path)

    write_markdown_matrix(all_results, md_path)
    print(f"\nMatrix JSON: {out_path}")
    print(f"Matrix MD:   {md_path}")


if __name__ == "__main__":
    main()

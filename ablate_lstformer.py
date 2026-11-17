"""
LST-Former fusion ablation study.

Investigates why the original concat+Dense fusion (Section 3.4.3) failed to
beat the standalone Transformer (see README Section 7). Trains LST-Former
with several alternative fusion strategies (src/models/lstformer_ae.py) on
the SAME fixed subset of users, then statistically compares each variant
against the already-trained Transformer/LSTM baselines from
outputs/results.json (must exist -- run `main.py --model all` first).

Usage:
    python ablate_lstformer.py --max-users 30
    python ablate_lstformer.py --fusion-types concat gated weighted_sum
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.evaluate import evaluate_user, measure_inference_latency
from src.preprocessing import load_all_users, sample_zero_effort_impostors
from src.train import build_model, reconstruction_errors, train_model


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Reproducibility -- see README Section 7.6 / main.py's set_seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

FUSION_TYPES = ["concat", "norm_concat", "gated", "weighted_sum", "hybrid_nobottleneck", "local_injection"]
ALPHA = 0.05


def paired_cohens_dz(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d_z = mean(diff) / sd(diff)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = diff.std(ddof=1)
    if sd == 0 or len(diff) < 2:
        return 0.0
    return float(diff.mean() / sd)


def run_variant(
    fusion_type: str,
    user_ids: list[str],
    all_users: dict,
    device: str,
    decoder_type: str = "lstm",
) -> dict:
    """Train+evaluate LST-Former with one fusion/decoder strategy across all user_ids."""
    results = []
    for uid in user_ids:
        splits = all_users[uid]
        train, val, test = splits.train, splits.val, splits.test
        if len(train) < config.BATCH_SIZE:
            results.append({"user": uid, "status": "skipped"})
            continue

        seq_len = train.shape[1]
        model = build_model("lstformer", seq_len=seq_len, fusion_type=fusion_type, decoder_type=decoder_type)
        history = train_model(model, train, val, save_path=None, device=device, model_name="lstformer")

        genuine_errors = reconstruction_errors(model, test, device).numpy()
        val_errors = reconstruction_errors(model, val, device).numpy()

        imposter_all = sample_zero_effort_impostors(uid, all_users, n_genuine_test=len(test))
        if imposter_all is None:
            results.append({"user": uid, "status": "skipped"})
            continue

        imposter_errors = reconstruction_errors(model, imposter_all, device).numpy()
        metrics = evaluate_user(genuine_errors, imposter_errors, val_errors)

        results.append({
            "user": uid,
            "status": "ok",
            "num_params": sum(p.numel() for p in model.parameters()),
            "best_val_loss": history["best_val_loss"],
            **metrics,
        })
    return {"fusion_type": fusion_type, "decoder_type": decoder_type, "per_user": results}


def paired_vs_baseline(variant_scores: dict, baseline_scores: dict, metric_name: str) -> dict | None:
    common = sorted(set(variant_scores) & set(baseline_scores))
    if len(common) < 2:
        return None
    a = np.array([baseline_scores[u] for u in common])
    b = np.array([variant_scores[u] for u in common])
    t_stat, p_value = stats.ttest_rel(b, a)
    d = paired_cohens_dz(b, a)
    return {
        "metric": metric_name,
        "n_users": len(common),
        "baseline_mean": float(np.mean(a)),
        "variant_mean": float(np.mean(b)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_at_0.05": bool(p_value < ALPHA),
        "cohens_d": d,
        "cohens_d_estimator": "paired_dz",
    }


def main():
    parser = argparse.ArgumentParser(description="LST-Former fusion ablation study")
    parser.add_argument("--max-users", type=int, default=30)
    parser.add_argument(
        "--users-file",
        type=Path,
        default=None,
        help="Optional text file of user IDs (one per line). Overrides --max-users "
        "prefix selection. Use outputs/common_users_81.txt for Experiment A.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Override sliding-window length (default: config.WINDOW_SIZE). "
        "Pass 25/50/100 to align with window_stability cells.",
    )
    parser.add_argument("--fusion-types", type=str, nargs="+", default=FUSION_TYPES)
    parser.add_argument("--decoder-types", type=str, nargs="+", default=["lstm"], choices=["lstm", "transformer"])
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=config.OUTPUT_DIR / "results.json",
        help="Existing results.json with lstm/transformer baselines to compare against",
    )
    parser.add_argument(
        "--baseline-from-window-stability",
        action="store_true",
        help="Pull lstm/transformer baselines from outputs/window_stability.json "
        "at --window-size (requires --window-size). Use for apples-to-apples "
        "Experiment A comparison against the N=25/50/100 sweep.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path (default: outputs/lstformer_ablation.json)",
    )
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    window_size = args.window_size or config.WINDOW_SIZE

    # --- baselines ---
    baseline_by_metric = {"lstm": {"eer": {}, "auc": {}}, "transformer": {"eer": {}, "auc": {}}}
    if args.baseline_from_window_stability:
        ws_path = config.OUTPUT_DIR / "window_stability.json"
        if not ws_path.exists():
            print(f"Missing {ws_path}. Run window_stability.py first.")
            sys.exit(1)
        if args.window_size is None:
            print("--baseline-from-window-stability requires --window-size")
            sys.exit(1)
        ws = json.loads(ws_path.read_text())
        for model_name in ("lstm", "transformer"):
            cell = ws.get(f"{model_name}_N{window_size}")
            if not cell:
                print(f"Missing cell {model_name}_N{window_size} in window_stability.json")
                sys.exit(1)
            for metric in ("eer", "auc"):
                baseline_by_metric[model_name][metric] = {
                    r["user"]: r[metric]
                    for r in cell["per_user"]
                    if r.get("status") == "ok" and metric in r
                }
        print(f"Baselines loaded from window_stability.json at N={window_size}")
    else:
        if not args.baseline_results.exists():
            print(f"Missing {args.baseline_results}. Run `python main.py --model all` first.")
            sys.exit(1)
        with open(args.baseline_results) as f:
            baseline_results = json.load(f)
        for model_name in ("lstm", "transformer"):
            per_user = baseline_results.get(model_name, {}).get("per_user", [])
            baseline_by_metric[model_name] = {
                metric: {r["user"]: r[metric] for r in per_user if r.get("status") == "ok" and metric in r}
                for metric in ("eer", "auc")
            }

    print(f"Device: {device}")
    print(f"Window size: {window_size}")
    print(f"Loading Clarkson II dataset from {config.DATA_DIR} ...")
    all_users = load_all_users(config.DATA_DIR, window_size=window_size)

    if args.users_file:
        requested = [
            line.strip() for line in args.users_file.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        user_ids = [u for u in requested if u in all_users]
        missing = [u for u in requested if u not in all_users]
        print(f"Using {len(user_ids)} users from {args.users_file.name} "
              f"({len(missing)} missing at N={window_size})")
    else:
        user_ids = list(all_users.keys())[: args.max_users]
        print(f"Using {len(user_ids)} users for the ablation (fixed subset, same for every variant)")

    out_path = args.output or (config.OUTPUT_DIR / "lstformer_ablation.json")
    all_variant_results = {}
    if out_path.exists():
        # resume-safe merge
        all_variant_results = json.loads(out_path.read_text())

    for decoder_type in args.decoder_types:
        for fusion_type in args.fusion_types:
            set_seed()  # each variant starts from the same RNG state -- fair, reproducible
            variant_name = fusion_type if decoder_type == "lstm" else f"{fusion_type}+{decoder_type}_decoder"
            if variant_name in all_variant_results and all_variant_results[variant_name].get("per_user"):
                n_done = len(all_variant_results[variant_name]["per_user"])
                if n_done >= len(user_ids):
                    print(f"\nSkipping {variant_name} (already complete, {n_done} users)")
                    continue
            print(f"\n{'=' * 60}\nVariant: {variant_name}\n{'=' * 60}")
            variant = run_variant(fusion_type, user_ids, all_users, device, decoder_type=decoder_type)
            all_variant_results[variant_name] = variant

            ok = [r for r in variant["per_user"] if r["status"] == "ok"]
            if not ok:
                print("  No valid users for this variant.")
                continue
            mean_eer = float(np.mean([r["eer"] for r in ok]))
            mean_auc = float(np.mean([r["auc"] for r in ok]))
            print(f"  n={len(ok)} | mean EER={mean_eer:.4f} | mean AUC={mean_auc:.4f}")

            variant_eer = {r["user"]: r["eer"] for r in ok}
            variant_auc = {r["user"]: r["auc"] for r in ok}
            for baseline_name in ("lstm", "transformer"):
                cmp_eer = paired_vs_baseline(variant_eer, baseline_by_metric[baseline_name]["eer"], "eer")
                cmp_auc = paired_vs_baseline(variant_auc, baseline_by_metric[baseline_name]["auc"], "auc")
                if cmp_eer:
                    sig = "significant" if cmp_eer["significant_at_0.05"] else "not significant"
                    print(
                        f"  vs {baseline_name}: EER {cmp_eer['baseline_mean']:.4f} -> "
                        f"{cmp_eer['variant_mean']:.4f} (p={cmp_eer['p_value']:.4f}, {sig}, d={cmp_eer['cohens_d']:.3f})"
                    )
                all_variant_results[variant_name].setdefault("comparisons", {})[f"vs_{baseline_name}"] = {
                    "eer": cmp_eer,
                    "auc": cmp_auc,
                }

            # incremental save
            config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(all_variant_results, f, indent=2)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_variant_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    md_path = out_path.with_suffix(".md")
    save_markdown_report(all_variant_results, md_path)
    print(f"Report saved to {md_path}")


def save_markdown_report(all_variant_results: dict, out_path: Path) -> None:
    lines = [
        "# LST-Former Fusion Ablation Study",
        "",
        "Investigates alternative fusion strategies for combining the LSTM and",
        "Transformer branches, since the original concat+Dense fusion did not",
        "outperform a standalone Transformer (see README Section 7).",
        "",
        "| Fusion type | n | Mean EER | Mean AUC | vs LSTM (EER, p) | vs Transformer (EER, p) |",
        "|---|---|---|---|---|---|",
    ]
    for fusion_type, data in all_variant_results.items():
        ok = [r for r in data["per_user"] if r["status"] == "ok"]
        if not ok:
            continue
        mean_eer = np.mean([r["eer"] for r in ok])
        mean_auc = np.mean([r["auc"] for r in ok])
        cmp = data.get("comparisons", {})
        lstm_cmp = cmp.get("vs_lstm", {}).get("eer")
        trans_cmp = cmp.get("vs_transformer", {}).get("eer")
        lstm_str = f"{lstm_cmp['p_value']:.4f}" if lstm_cmp else "n/a"
        trans_str = f"{trans_cmp['p_value']:.4f}" if trans_cmp else "n/a"
        lines.append(
            f"| {fusion_type} | {len(ok)} | {mean_eer:.4f} | {mean_auc:.4f} | "
            f"p={lstm_str} | p={trans_str} |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

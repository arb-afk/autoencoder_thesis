"""
Comparative Biometric Statistical Analysis (Section 3.10.1).

Reads outputs/results.json (produced by main.py with --model all, run over
the SAME set of users for every model) and, for each baseline vs. the
proposed LST-Former, computes:

  - A paired-sample t-test on per-user EER (H0: means are not significantly
    different; alpha = 0.05)
  - Cohen's d effect size, to judge whether any statistically significant
    difference is also practically meaningful

Usage:
    python main.py --model all --max-users 20      # generate outputs/results.json
    python statistical_analysis.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config

ALPHA = 0.05


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples using the pooled standard deviation."""
    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(
        ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)) / (n_a + n_b - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def interpret_d(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def load_metric_by_user(results: dict, model_name: str, metric: str) -> dict[str, float]:
    per_user = results[model_name]["per_user"]
    return {r["user"]: r[metric] for r in per_user if r.get("status") == "ok" and metric in r}


def paired_comparison(baseline_scores: dict, proposed_scores: dict, metric: str) -> dict | None:
    """Paired t-test + Cohen's d over users present in both score dicts."""
    common_users = sorted(set(baseline_scores) & set(proposed_scores))
    if len(common_users) < 2:
        return None

    a = np.array([baseline_scores[u] for u in common_users])  # baseline
    b = np.array([proposed_scores[u] for u in common_users])  # LST-Former

    t_stat, p_value = stats.ttest_rel(a, b)
    d = cohens_d(b, a)  # positive d => LST-Former has higher metric than baseline

    return {
        "metric": metric,
        "n_users": len(common_users),
        "baseline_mean": float(np.mean(a)),
        "proposed_mean": float(np.mean(b)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_at_0.05": bool(p_value < ALPHA),
        "cohens_d": d,
        "effect_size": interpret_d(d),
    }


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis (Section 3.10.1)")
    parser.add_argument(
        "--results",
        type=Path,
        default=config.OUTPUT_DIR / "results.json",
        help="Path to results.json produced by main.py --model all",
    )
    parser.add_argument("--metric", type=str, default="eer", choices=["eer", "auc", "far", "frr"])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the JSON summary (default: <results_dir>/statistical_analysis.json)",
    )
    args = parser.parse_args()

    if not args.results.exists():
        print(f"Could not find {args.results}.")
        print("Run: python main.py --model all --max-users <N>  first, over the SAME users.")
        sys.exit(1)

    with open(args.results) as f:
        results = json.load(f)

    for required in ("lstm", "transformer", "lstformer"):
        if required not in results:
            print(f"Missing '{required}' in {args.results}; re-run main.py --model all")
            sys.exit(1)

    proposed_scores = load_metric_by_user(results, "lstformer", args.metric)

    print(f"\nPaired-sample t-test on {args.metric.upper()} (LST-Former vs. each baseline)")
    print(f"H0: means are not significantly different | alpha = {ALPHA}")
    print("=" * 75)

    summary = {}
    for baseline in ("lstm", "transformer"):
        baseline_scores = load_metric_by_user(results, baseline, args.metric)
        comparison = paired_comparison(baseline_scores, proposed_scores, args.metric)
        summary[f"lstformer_vs_{baseline}"] = comparison

        label = "LSTM (Baseline)" if baseline == "lstm" else "Transformer (Challenger)"
        print(f"\nLST-Former vs. {label}")
        if comparison is None:
            print("  Not enough overlapping users with valid results to compare.")
            continue
        print(f"  n = {comparison['n_users']} users")
        print(f"  Mean {args.metric.upper()}: baseline={comparison['baseline_mean']:.4f}  "
              f"LST-Former={comparison['proposed_mean']:.4f}")
        print(f"  t = {comparison['t_statistic']:.4f}, p = {comparison['p_value']:.4f} "
              f"({'significant' if comparison['significant_at_0.05'] else 'not significant'} at alpha=0.05)")
        print(f"  Cohen's d = {comparison['cohens_d']:.4f} ({comparison['effect_size']} effect)")

    out_path = args.output or (args.results.parent / f"statistical_analysis_{args.metric}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

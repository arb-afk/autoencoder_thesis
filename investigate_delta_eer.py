"""
Experiment B — per-user ΔEER analysis (advisor investigation).

For each window length N in the window-stability sweep, compute:

    DeltaEER_u = EER_Transformer(u) - EER_LST-Former(u)

Positive Delta => LST-Former better for that user. Reports whether the mean
advantage is broad across users or driven by a small subset.

Also dumps per-user recon-gap / AUC / FAR / FRR deltas for inspection.

Usage:
    python investigate_delta_eer.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
WS = json.loads((OUT / "window_stability.json").read_text())
ANALYSIS = json.loads((OUT / "window_stability_analysis.json").read_text())

LENGTHS = [25, 50, 100]
ALPHA = 0.05


def ok_users(cell_key: str) -> dict[str, dict]:
    return {
        r["user"]: r
        for r in WS[cell_key]["per_user"]
        if r.get("status") == "ok"
    }


def summarize_deltas(deltas: np.ndarray, label: str) -> dict:
    """Describe the distribution of per-user improvements."""
    n = len(deltas)
    pos = int(np.sum(deltas > 0))
    neg = int(np.sum(deltas < 0))
    zero = int(np.sum(deltas == 0))
    # "driven by subset?" — share of total positive mass from top 10%/20%
    pos_only = deltas[deltas > 0]
    if len(pos_only) == 0:
        top10_share = top20_share = 0.0
    else:
        sorted_pos = np.sort(pos_only)[::-1]
        total = sorted_pos.sum()
        k10 = max(1, int(np.ceil(0.10 * len(sorted_pos))))
        k20 = max(1, int(np.ceil(0.20 * len(sorted_pos))))
        top10_share = float(sorted_pos[:k10].sum() / total) if total > 0 else 0.0
        top20_share = float(sorted_pos[:k20].sum() / total) if total > 0 else 0.0

    t, p = stats.ttest_1samp(deltas, 0.0)
    # Wilcoxon signed-rank (more robust to outliers / non-normality)
    try:
        w_stat, w_p = stats.wilcoxon(deltas, alternative="greater")
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    return {
        "label": label,
        "n": n,
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "std_delta": float(deltas.std(ddof=1)),
        "pct_lstformer_better": 100.0 * pos / n,
        "pct_transformer_better": 100.0 * neg / n,
        "pct_tied": 100.0 * zero / n,
        "n_lstformer_better": pos,
        "n_transformer_better": neg,
        "top10pct_of_winners_share_of_positive_mass": top10_share,
        "top20pct_of_winners_share_of_positive_mass": top20_share,
        "ttest_mean_gt_0": {"t": float(t), "p": float(p), "significant": bool(p < ALPHA)},
        "wilcoxon_delta_gt_0": {
            "statistic": float(w_stat),
            "p": float(w_p),
            "significant": bool(w_p < ALPHA) if w_p == w_p else False,
        },
        "quartiles": {
            "q25": float(np.percentile(deltas, 25)),
            "q50": float(np.percentile(deltas, 50)),
            "q75": float(np.percentile(deltas, 75)),
        },
    }


def main():
    # Prefer the analysis file's common-user set if present; else recompute
    common = None
    for m in ("lstm", "transformer", "lstformer"):
        for n in LENGTHS:
            users = set(ok_users(f"{m}_N{n}"))
            common = users if common is None else (common & users)
    common = sorted(common)
    print(f"Common users (all 9 cells): {len(common)}")

    report: dict = {"common_users": common, "n_common": len(common), "by_length": {}}
    lines = [
        "# Experiment B -- Per-user Delta EER (Transformer - LST-Former)",
        "",
        "Positive Delta EER => LST-Former better for that user.",
        f"Restricted to the **n={len(common)}** users present in every model x N cell.",
        "",
    ]

    for n in LENGTHS:
        t_map = ok_users(f"transformer_N{n}")
        l_map = ok_users(f"lstformer_N{n}")
        deltas = np.array([t_map[u]["eer"] - l_map[u]["eer"] for u in common])
        auc_d = np.array([l_map[u]["auc"] - t_map[u]["auc"] for u in common])  # higher better
        gap_d = np.array([
            l_map[u].get("recon_gap", l_map[u]["imposter_mean"] - l_map[u]["genuine_mean"])
            - t_map[u].get("recon_gap", t_map[u]["imposter_mean"] - t_map[u]["genuine_mean"])
            for u in common
        ])

        summary = summarize_deltas(deltas, f"N={n}")
        report["by_length"][str(n)] = {
            "eer_delta": summary,
            "mean_auc_delta_lstformer_minus_transformer": float(auc_d.mean()),
            "mean_recon_gap_delta_lstformer_minus_transformer": float(gap_d.mean()),
            "per_user": [
                {
                    "user": u,
                    "delta_eer": float(t_map[u]["eer"] - l_map[u]["eer"]),
                    "transformer_eer": float(t_map[u]["eer"]),
                    "lstformer_eer": float(l_map[u]["eer"]),
                    "transformer_auc": float(t_map[u]["auc"]),
                    "lstformer_auc": float(l_map[u]["auc"]),
                    "transformer_recon_gap": float(
                        t_map[u].get("recon_gap", t_map[u]["imposter_mean"] - t_map[u]["genuine_mean"])
                    ),
                    "lstformer_recon_gap": float(
                        l_map[u].get("recon_gap", l_map[u]["imposter_mean"] - l_map[u]["genuine_mean"])
                    ),
                }
                for u in common
            ],
        }

        s = summary
        lines += [
            f"## N = {n}",
            "",
            f"- Mean Delta EER = **{s['mean_delta']:.4f}** (median {s['median_delta']:.4f})",
            f"- LST-Former better for **{s['pct_lstformer_better']:.1f}%** of users "
            f"({s['n_lstformer_better']}/{s['n']})",
            f"- Transformer better for **{s['pct_transformer_better']:.1f}%** "
            f"({s['n_transformer_better']}/{s['n']})",
            f"- Top 10% of winners account for **{100*s['top10pct_of_winners_share_of_positive_mass']:.1f}%** "
            f"of total positive Delta EER mass; top 20% -> "
            f"**{100*s['top20pct_of_winners_share_of_positive_mass']:.1f}%**",
            f"- One-sample t-test (mean Delta > 0): p={s['ttest_mean_gt_0']['p']:.4g} "
            f"({'sig' if s['ttest_mean_gt_0']['significant'] else 'n.s.'})",
            f"- Wilcoxon (Delta > 0): p={s['wilcoxon_delta_gt_0']['p']:.4g} "
            f"({'sig' if s['wilcoxon_delta_gt_0']['significant'] else 'n.s.'})",
            f"- Mean AUC Delta (LST-F - Trans) = {float(auc_d.mean()):.4f}",
            f"- Mean recon-gap Delta (LST-F - Trans) = {float(gap_d.mean()):.4f}",
            "",
        ]

        # Interpretation hint
        if s["top20pct_of_winners_share_of_positive_mass"] > 0.55:
            hint = (
                "Advantage looks **concentrated** in a subset of users "
                "(top quintile of winners carries >55% of positive mass)."
            )
        else:
            hint = (
                "Advantage looks **relatively broad** across users "
                "(positive mass not dominated by a small winner subset)."
            )
        lines.append(f"**Reading:** {hint}")
        lines.append("")

    lines += [
        "## Bottom line for the thesis",
        "",
        "If % of users helped is high *and* top-quintile share of positive mass",
        "is moderate, the N=25/50 significance is a **broad** effect -- safer to",
        "claim as architectural. If most users are near zero and a few large",
        "winners drive the mean, frame the result as **heterogeneous / user-",
        "dependent**, not a uniform superiority claim.",
        "",
    ]

    out_json = OUT / "delta_eer_analysis.json"
    out_md = OUT / "delta_eer_analysis.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")

    # Also dump the common-user list for Experiment A
    users_path = OUT / "common_users_81.txt"
    users_path.write_text("\n".join(common) + "\n", encoding="utf-8")
    print(f"Wrote {users_path} for Experiment A")


if __name__ == "__main__":
    main()

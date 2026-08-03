"""Post-process window_stability.json: paired trajectories + merged efficiency."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
WS = json.load(open(OUT / "window_stability.json"))
RS = json.load(open(OUT / "resource_scaling.json"))

MODELS = ["lstm", "transformer", "lstformer"]
LENGTHS = [25, 50, 100]
ALPHA = 0.05


def ok_map(cell_key: str, metric: str) -> dict[str, float]:
    return {
        r["user"]: r[metric]
        for r in WS[cell_key]["per_user"]
        if r.get("status") == "ok" and metric in r
    }


def paired(a: dict[str, float], b: dict[str, float], metric: str) -> dict:
    users = sorted(set(a) & set(b))
    xa = np.array([a[u] for u in users], dtype=float)
    xb = np.array([b[u] for u in users], dtype=float)
    t, p = stats.ttest_rel(xa, xb)
    diff = xb - xa
    d = float(diff.mean() / diff.std(ddof=1)) if len(users) > 1 and diff.std(ddof=1) > 0 else 0.0
    return {
        "metric": metric,
        "n_users": len(users),
        "a_mean": float(xa.mean()),
        "b_mean": float(xb.mean()),
        "t": float(t),
        "p": float(p),
        "cohens_d": d,
        "significant": bool(p < ALPHA),
    }


# Intersection of users who succeed at EVERY model × EVERY length
common = None
for m in MODELS:
    for n in LENGTHS:
        users = set(ok_map(f"{m}_N{n}", "eer"))
        common = users if common is None else (common & users)
common = sorted(common)
print(f"Common users across all 9 cells: {len(common)}")

matrix = {"common_users": len(common), "lengths": LENGTHS, "models": MODELS, "cells": {}, "paired": {}, "trajectory": {}}

for m in MODELS:
    for n in LENGTHS:
        key = f"{m}_N{n}"
        s = WS[key]["summary"]
        # efficiency from resource_scaling when available
        rs_raw = RS.get("raw", {}).get(m, {})
        lat = mem = params = None
        if rs_raw and n in rs_raw.get("N", []):
            i = rs_raw["N"].index(n)
            lat = rs_raw["latency_ms"][i]
            mem = rs_raw["memory_mb"][i]
            params = rs_raw["params"][i]
        eer_c = [ok_map(key, "eer")[u] for u in common]
        auc_c = [ok_map(key, "auc")[u] for u in common]
        matrix["cells"][key] = {
            **{k: s.get(k) for k in (
                "users_evaluated", "mean_eer", "mean_auc", "mean_far", "mean_frr",
                "mean_genuine_recon", "mean_imposter_recon", "mean_recon_gap",
                "mean_latency_ms", "mean_peak_vram_mb", "mean_params",
            )},
            "common_mean_eer": float(np.mean(eer_c)),
            "common_mean_auc": float(np.mean(auc_c)),
            "resource_scaling_latency_ms": lat,
            "resource_scaling_memory_mb": mem,
            "resource_scaling_params": params,
        }

# Paired LST-Former vs each baseline at each N (common users)
for n in LENGTHS:
    lf_eer = {u: ok_map(f"lstformer_N{n}", "eer")[u] for u in common}
    lf_auc = {u: ok_map(f"lstformer_N{n}", "auc")[u] for u in common}
    for base in ("lstm", "transformer"):
        be = {u: ok_map(f"{base}_N{n}", "eer")[u] for u in common}
        ba = {u: ok_map(f"{base}_N{n}", "auc")[u] for u in common}
        matrix["paired"][f"lstformer_vs_{base}_N{n}_eer"] = paired(be, lf_eer, "eer")
        matrix["paired"][f"lstformer_vs_{base}_N{n}_auc"] = paired(ba, lf_auc, "auc")
        # note: paired() does ttest_rel(a,b) with a=baseline, b=lstformer
        # for EER lower is better so negative (b-a) means LST-Former better

# Trajectory: delta EER from N=25 to N=100 on common users
for m in MODELS:
    e25 = np.array([ok_map(f"{m}_N25", "eer")[u] for u in common])
    e100 = np.array([ok_map(f"{m}_N100", "eer")[u] for u in common])
    t, p = stats.ttest_rel(e25, e100)
    matrix["trajectory"][m] = {
        "eer_N25": float(e25.mean()),
        "eer_N50": float(np.mean([ok_map(f"{m}_N50", "eer")[u] for u in common])),
        "eer_N100": float(e100.mean()),
        "delta_eer_25_to_100": float(e100.mean() - e25.mean()),
        "paired_t_25_vs_100": float(t),
        "p_25_vs_100": float(p),
        "improves_with_longer_window": bool(e100.mean() < e25.mean() and p < ALPHA),
    }

out_json = OUT / "window_stability_analysis.json"
with open(out_json, "w") as f:
    json.dump(matrix, f, indent=2)

# Markdown write-up
lines = [
    "# Architecture × Window-Length Stability Matrix (final)",
    "",
    "Source: `window_stability.json` (seeded PyTorch, `RANDOM_SEED=42`).",
    f"Paired analyses use the **intersection of {len(common)} users** who succeed",
    "at every architecture × every window length (25/50/100).",
    "",
    "## Mean EER trajectory (lower is better)",
    "",
    "| Window N | LSTM | Transformer | LST-Former | LST-F − Trans |",
    "| ---: | ---: | ---: | ---: | ---: |",
]
for n in LENGTHS:
    row = [str(n)]
    vals = []
    for m in MODELS:
        v = matrix["cells"][f"{m}_N{n}"]["common_mean_eer"]
        vals.append(v)
        row.append(f"{v:.4f}")
    row.append(f"{vals[2]-vals[1]:+.4f}")
    lines.append("| " + " | ".join(row) + " |")

lines += [
    "",
    "## Mean AUC trajectory (higher is better)",
    "",
    "| Window N | LSTM | Transformer | LST-Former |",
    "| ---: | ---: | ---: | ---: |",
]
for n in LENGTHS:
    row = [str(n)]
    for m in MODELS:
        row.append(f"{matrix['cells'][f'{m}_N{n}']['common_mean_auc']:.4f}")
    lines.append("| " + " | ".join(row) + " |")

lines += [
    "",
    "## Does longer context help? (paired EER N=25 vs N=100, common users)",
    "",
    "| Model | EER@25 | EER@100 | Δ | p | Improves? |",
    "|---|---:|---:|---:|---:|---|",
]
for m in MODELS:
    t = matrix["trajectory"][m]
    lines.append(
        f"| {m} | {t['eer_N25']:.4f} | {t['eer_N100']:.4f} | "
        f"{t['delta_eer_25_to_100']:+.4f} | {t['p_25_vs_100']:.4f} | "
        f"{'yes' if t['improves_with_longer_window'] else 'no'} |"
    )

lines += [
    "",
    "## LST-Former vs baselines at each N (paired, common users)",
    "",
    "| N | vs | Metric | baseline | LST-Former | p | sig? | d |",
    "| ---: | --- | --- | ---: | ---: | ---: | --- | ---: |",
]
for n in LENGTHS:
    for base in ("lstm", "transformer"):
        for metric in ("eer", "auc"):
            p = matrix["paired"][f"lstformer_vs_{base}_N{n}_{metric}"]
            # p['a_mean']=baseline, p['b_mean']=lstformer
            lines.append(
                f"| {n} | {base} | {metric} | {p['a_mean']:.4f} | {p['b_mean']:.4f} | "
                f"{p['p']:.4f} | {'yes' if p['significant'] else 'no'} | {p['cohens_d']:+.3f} |"
            )

lines += [
    "",
    "## Reconstruction-error gap (imposter − genuine mean)",
    "",
    "| Window N | LSTM | Transformer | LST-Former |",
    "| ---: | ---: | ---: | ---: |",
]
for n in LENGTHS:
    row = [str(n)]
    for m in MODELS:
        row.append(f"{WS[f'{m}_N{n}']['summary']['mean_recon_gap']:.4f}")
    lines.append("| " + " | ".join(row) + " |")

lines += [
    "",
    "## Efficiency (resource_scaling.py reference measurements)",
    "",
    "| N | Model | latency_ms | memory_mb | params |",
    "| ---: | --- | ---: | ---: | ---: |",
]
for n in LENGTHS:
    for m in MODELS:
        c = matrix["cells"][f"{m}_N{n}"]
        lat = c["resource_scaling_latency_ms"]
        mem = c["resource_scaling_memory_mb"]
        params = c["resource_scaling_params"]
        if lat is None:
            lines.append(f"| {n} | {m} | — | — | — |")
        else:
            lines.append(f"| {n} | {m} | {lat:.3f} | {mem:.1f} | {int(params):,} |")

lines += [
    "",
    "## What the data says (for Results / Discussion)",
    "",
    "1. **LST-Former has the best mean EER at every tested window length**",
    "   (25, 50, 100) on the common-user set — not only at long N.",
    "2. **Both attention-based models improve as N grows; LSTM does not.**",
    "   That supports the thesis claim that sequence length challenges",
    "   architectures differently: longer temporal context helps",
    "   Transformer/LST-Former classification stability more than LSTM.",
    "3. **The LST-Former − Transformer EER gap shrinks as N grows**",
    "   (largest at short N, smaller at N=100). That is *not* the",
    "   “hybrid only helps at long windows” story — if anything, the",
    "   parallel branch’s relative edge is clearer when context is short,",
    "   while at long N the standalone Transformer largely catches up.",
    "4. **Efficiency remains the clear tradeoff:** LST-Former is ~2–4×",
    "   slower and more parameterized than Transformer at the same N",
    "   (see resource_scaling). Security gains must be weighed against cost.",
    "5. Frame honestly: do **not** claim inherent superiority. Claim that",
    "   under unconstrained Clarkson II HT/FT windows, parallel local–global",
    "   fusion provided a **measurable, length-dependent** advantage over",
    "   single-branch baselines, with Transformer remaining the closest",
    "   competitor and LSTM remaining clearly weaker.",
    "",
]

(OUT / "window_stability.md").write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {out_json}")
print(f"Wrote {OUT / 'window_stability.md'}")
for m in MODELS:
    print("traj", m, matrix["trajectory"][m])

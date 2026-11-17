# Leak-free protocol rerun — results summary

**Protocol:** chronological feature split → train-only z-score → windows inside each partition; fixed per-user impostor RNG.  
**Population:** n=81 common users (unchanged count).  
**Artifacts:** `window_stability.json`, `window_stability.md`, `window_stability_analysis.json`, `delta_eer_analysis.md`, `lstformer_ablation_aligned_N50.md`  
**Backups:** `outputs/pre_leakfree_*`

## Primary mean EER (n=81)

| N | LSTM | Transformer | LST-Former | vs Trans (p, d) |
|---:|---:|---:|---:|---|
| 25 | 0.4444 | 0.3521 | **0.3342** | p=0.0746, d=−0.201 (**n.s.**) |
| 50 | 0.4626 | 0.3447 | **0.3264** | p=0.1155, d=−0.177 (**n.s.**) |
| 100 | 0.4609 | 0.3350 | **0.3233** | p=0.2471, d=−0.130 (**n.s.**) |

vs LSTM: still large/significant at all N (d ≈ −0.80 to −1.10, p≈0).

## Primary mean AUC (n=81)

| N | LSTM | Transformer | LST-Former | vs Trans (p, d) |
|---:|---:|---:|---:|---|
| 25 | 0.5288 | 0.6669 | **0.6969** | p=0.0191, d=+0.266 (sig, small) |
| 50 | 0.5075 | 0.6785 | **0.7127** | p=0.0156, d=+0.274 (sig, small) |
| 100 | 0.5040 | 0.6883 | **0.7014** | p=0.2879, d=+0.119 (n.s.) |

## vs previous (leaky) primary story

| Claim | Old (leaky) | New (leak-free) |
|---|---|---|
| LST-F EER @25/50/100 | 0.3271 / **0.311** / 0.3182 | 0.3342 / **0.3264** / 0.3233 |
| vs Trans EER sig @25/50 | yes (p=0.003 / 0.014) | **no** (p=0.075 / 0.116) |
| vs Trans AUC sig @25/50 | yes | yes (weaker) |
| vs LSTM | large win | large win (unchanged story) |

**Thesis implication:** Do **not** keep abstract/results that say primary EER vs Transformer is significant at N=25/50. Prefer: clear win vs LSTM; small, often non-significant EER edge vs Transformer; AUC still shows a small significant edge at N=25/50; N=100 catch-up remains.

## ΔEER heterogeneity (Transformer − LST-Former)

| N | Mean Δ | Hybrid better | t-test p | Wilcoxon p |
|---:|---:|---|---:|---:|
| 25 | 0.0179 | 58.0% (47/81) | 0.075 n.s. | 0.025 sig |
| 50 | 0.0182 | 63.0% (51/81) | 0.116 n.s. | 0.025 sig |
| 100 | 0.0117 | 53.1% (43/81) | 0.247 n.s. | 0.099 n.s. |

## Aligned fusion ablation (N=50, n=81, separate run)

| Fusion | Mean EER | Mean AUC | vs Trans EER p |
|---|---:|---:|---:|
| concat | 0.4581 | 0.5113 | ~0 (worse) |
| local_injection | 0.3547 | 0.6553 | 0.359 |
| hybrid_nobottleneck | **0.3228** | **0.7033** | 0.052 |

Primary window-stability hybrid @N=50 remains **0.3264** (do not swap with ablation 0.3228).

## Do not use without cleanup

Mean reconstruction-error gaps in `window_stability.md` are numerically extreme under train-only z-score (outlier MSE). Prefer EER/AUC for Results; treat recon gaps as unreliable until clipped/median-summarized.

# Canonical experimental protocol (LOCKED for thesis writing)

**Status: LOCKED — stop unplanned experiments.** Primary Results tables use
this protocol only. Do not cherry-pick n=88, do not re-run exploratory
ablations for the main claim, and do not invent new architectures until
the Results chapter is drafted around this matrix.

This document resolves the 27 / 81 / 88 user-count confusion and states
exactly which results are **primary** vs **exploratory**.

Last updated after Experiments A/B and the window-stability sweep;
advisor-confirmed as the canonical population/protocol.

---

## 1. Why the numbers differ

| Label | n | What it is |
|---|---:|---|
| Main `results.json` | **88** | Users with enough windows at **N=50 only** (default `WINDOW_SIZE`) |
| Window-stability cell at N=50 | **88** | Same ceiling as above |
| Window-stability cell at N=100 | **84** | Longer windows → fewer users reach `BATCH_SIZE` |
| **Canonical common set** | **81** | Intersection of users who succeed for **all 3 models at N=25, 50, and 100** |
| Early fusion ablation | **~27** | Exploratory subset (`--max-users 30`); **not** for thesis claims |

The 81-user list is fixed in `outputs/common_users_81.txt`.

Users present at N=50 for all models but **not** in the canonical 81 are those who fail at N=25 and/or N=100 for at least one architecture (insufficient windows / skipped).

---

## 2. Canonical population and protocol (PRIMARY)

**Population:** the **n=81** common-user set.

**Why this one:** It is the only set that supports fair paired comparisons
*across window lengths*. Using n=88 for N=50 and a different n for N=100
would mix populations and invalidate the architecture × window trajectory.

**Protocol:**

| Item | Setting |
|---|---|
| Dataset | Clarkson II filtered (unconstrained) |
| Features | Hold Time + Flight Time, per-user z-score |
| Window lengths | N ∈ {25, 50, 100}, stride = N/2 (50% overlap) |
| Split | Chronological 70/10/20 |
| Models | LSTM-AE, Transformer-AE, LST-Former (`hybrid_nobottleneck`) |
| Seed | `RANDOM_SEED = 42` |
| Device (train) | CUDA; latency always CPU |
| Unit of analysis | Per-user metrics; paired tests on the same 81 IDs |
| Primary artifacts | `outputs/window_stability.json`, `window_stability_analysis.json`, `delta_eer_analysis.json` |

**Primary tables to cite in Results:**

### Architecture × window EER (n=81)

| N | LSTM | Transformer | LST-Former | vs Trans (p, Cohen's d) | vs LSTM (p, d) |
|---:|---:|---:|---:|---|---|
| 25 | 0.440 | 0.370 | **0.327** | p=0.003, d=−0.34 (small) | p≈0, d=−0.79 (large) |
| 50 | 0.451 | 0.345 | **0.311** | p=0.014, d=−0.28 (small) | p≈0, d=−0.91 (large) |
| 100 | 0.455 | 0.332 | **0.318** | p=0.302, d=−0.12 (negligible) | p≈0, d=−0.93 (large) |

### Locked interpretation (use this wording)

> The LST-Former provides a measurable advantage over LSTM across the
> tested sequence lengths, while its advantage over Transformer is smaller
> and depends on sequence length and the manner in which the recurrent and
> self-attention representations are integrated. At longer windows, the
> Transformer catches up, suggesting that the additional recurrent
> representation does not provide a universal benefit.

Supporting points (same paragraph or Discussion):

- vs LSTM: large effect at all N (primary clear differentiation).
- vs Transformer: small significant edge at N=25/50; n.s. at N=100.
- Fusion: benefit is design-dependent (`hybrid_nobottleneck`), not automatic
  local+global fusion (`concat` fails; `local_injection` ties Transformer).
- Per-user: majority improve at short/medium N; magnitude heterogeneous.
- Cost: higher latency/memory → security–efficiency trade-off.

**Do not write:** “LST-Former is significantly / inherently better than
Transformer.” If mentioning a p-value vs Transformer, always pair it with
effect size (small) and the N=100 catch-up.

---

## 3. Secondary / exploratory (do NOT mix into primary tables)

| Artifact | Role |
|---|---|
| `outputs/results.json` + `statistical_analysis_*.json` (n=88, N=50) | Same protocol at single N; **supplementary**. Note: LST-F vs Trans p=0.046 with **d≈0.14 (negligible)** — illustrates “significant ≠ large.” |
| Early `lstformer_ablation.json` (n≈27) | Discovery only; superseded by aligned ablation |
| `lstformer_ablation_aligned_N50.json` (n=81) | Supports the **fusion-dependence** claim (primary *finding*, but fusion variants are not part of the main 3-model matrix) |
| TF CPU/GPU cross-checks (n=18) | Robustness appendix, not primary |
| Resource scaling / log-log exponents | Efficiency chapter; wording = **consistent with** O(N)/O(N²), not “proves” |

---

## 4. Fusion-dependence evidence (supports theory refinement)

Same 81 users, N=50, vs window-stability Transformer (EER 0.345):

| Fusion | Mean EER | vs Transformer |
|---|---:|---|
| concat | 0.448 | significantly worse |
| local_injection | 0.341 | tied (p=0.74) |
| hybrid_nobottleneck (window-stability) | 0.311 | p=0.014 (small effect) |

Refined claim:

> Benefits of LST-Former depend on how recurrent and self-attention
> representations are integrated, rather than simply on combining the two
> architectures.

---

## 5. What not to do next

- **Stop experimenting for now.** Lock this matrix and write Results.
- Do **not** cherry-pick n=88 when it looks better and n=81 when it looks safer.
- Do **not** run more unplanned ablations before drafting Results around this matrix.
- Experiment C (representation complementarity) remains optional Discussion
  depth only — **not** required for the locked contribution claim.

---

## 6. File map for the thesis writer

| Need | File |
|---|---|
| Primary EER/AUC × N matrix | `outputs/window_stability.md` + `window_stability_analysis.json` |
| ΔEER heterogeneity | `outputs/delta_eer_analysis.md` |
| Fusion dependence | `outputs/lstformer_ablation_aligned_N50.md` |
| Efficiency / scaling | `outputs/asymptotic_report.md`, `resource_scaling.json` |
| Narrative / wording | `outputs/thesis_framing_notes.md` |
| Fixed user IDs | `outputs/common_users_81.txt` |

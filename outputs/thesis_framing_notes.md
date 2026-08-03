# Thesis framing notes (post-proposal refinement)

These notes implement your advisor's feedback without inventing a new
methodology. Use them when revising Chapters 1–3 and the Results/Discussion
write-up. Evidence for the architecture × window matrix lives in
`outputs/window_stability.json` / `.md` once that sweep finishes.

---

## 0. Canonical protocol (read this first)

**Primary thesis population = n=81 users** who succeed for every model at
N=25, 50, and 100. Fixed ID list: `outputs/common_users_81.txt`.

Full lock-in document: **`outputs/CANONICAL_PROTOCOL.md`**.

- n=88 = N=50-only ceiling (supplementary; do not mix with N=100).
- n≈27 = early exploratory ablation (do not cite as primary).
- Prefer effect sizes with p-values; “significant” alone is not the claim.

**Locked abstract/conclusion wording:**

> The LST-Former provides a measurable advantage over LSTM across the
> tested sequence lengths, while its advantage over Transformer is smaller
> and depends on sequence length and the manner in which the recurrent and
> self-attention representations are integrated. At longer windows, the
> Transformer catches up, suggesting that the additional recurrent
> representation does not provide a universal benefit.

**Status: matrix locked — write Results; no further unplanned experiments.**


Your defended proposal already asks what happens when *architecture* changes
under the **same unconstrained keystroke stream**. Clarkson II is the right
environment for that question.

**Limitation wording (use this, don't apologize for it):**

> The dataset does not provide sufficient activity context to determine
> whether each observed keyboard interaction corresponds to text entry,
> gaming, navigation, or another keyboard-intensive activity. Therefore,
> the study evaluates authentication of observed keyboard behavior under
> unconstrained conditions rather than activity-specific typing
> authentication.

The lack of activity labels is part of the experimental environment, not a
hidden flaw you must paper over.

---

## 2. Reframe the research question (you already almost say this)

**Avoid implying the thesis is:** “LST-Former should beat Transformer.”

**Prefer (aligned with your stated objective):**

> Does combining sequential and self-attention mechanisms provide a
> measurable advantage over either mechanism alone when modeling
> unconstrained keystroke behavior?

Or, matching your methodology language:

> Whether parallel local–global feature fusion enhances classification
> stability relative to single-branch LSTM or Transformer autoencoders
> under identical preprocessing and sequence structures.

---

## 3. Soften causal language

Replace phrases like:

- “inherently superior”
- “is inherently superior”
- “achieve a significantly narrower reconstruction error gap” (when stated
  as a hypothesis presented as fact)

With:

- “whether parallel local–global feature fusion provides **measurable
  advantages** over either standalone architecture”
- “whether … improves … relative to …”
- “empirically compare …”

This is an empirical study, not a theorem proving LST-Former must win.

---

## 4. Strengthen the theoretical justification (the real gap)

Panel risk: “If the Transformer already attends locally and globally, what
does the LSTM branch add?”

Current paper claim (reasonable but incomplete):

| Branch | Claimed role |
|---|---|
| LSTM | Localized / micro-rhythmic patterns |
| Transformer | Global / macro-context |
| Parallel fusion | Both |

**Stronger framing to add (still testable, not overclaimed):**

1. **Inductive bias, not just capacity.** Self-attention *can* represent
   local structure, but it is not *biased* toward ordered causal
   accumulation the way an LSTM is. The LSTM branch supplies a
   directionally ordered local-rhythm prior; the Transformer supplies
   content-based long-range mixing. Fusion asks whether that prior still
   helps once HT/FT sequences are long enough for both mechanisms to matter.

2. **Redundancy is a scientific outcome.** If Transformer ≈ LST-Former
   across window lengths, a defensible interpretation is:
   > The Transformer already captures most of the information the LSTM
   > branch contributes on this feature representation; the parallel
   > branch is largely redundant under unconstrained HT/FT windows.

   That does **not** kill the thesis — it answers the fusion question.

3. **Where the hypothesis is most challenged:** longer windows (your
   existing sequence-length experiment). Short N → little room for
   local/global complementarity; long N → LSTM must carry information
   farther while attention still sees all positions. Make that sweep the
   Results centerpiece, not a side plot.

---

## 5. Centerpiece experiment (already in Chapter 3 — expand it)

Do **not** invent Buffalo-style methodology. Expand the planned question:

> How does the length of input sequences affect reconstruction-error
> distributions and classification stability?

Report a matrix (script: `window_stability.py`):

| Window N | LSTM EER/AUC/… | Transformer … | LST-Former … | latency / VRAM / params |

Then narrate **trajectories**, not a single N=50 horse race:

- LST-Former only helps at long N → supports complementarity under longer
  temporal context.
- Flat Transformer ≈ LST-Former at every N → supports redundancy finding.
- Efficiency split with tied security → still an architectural result.

Paired comparisons across N should use the **intersection of users** who
succeed at every length (longer N skips more users).

---

## 6. Technical inconsistencies to fix in the PDF

| Issue | Fix |
|---|---|
| Scope says PyTorch; Ch. 3 says TensorFlow 2.16.1 + Keras | Pick one primary framework (recommend PyTorch for reported GPU numbers; keep TF as cross-check appendix, or vice versa — but say so explicitly). |
| LSTM decoder text saying TimeDistributed Dense “uses the self-attention mechanism” | Remove — standalone LSTM baseline must not claim attention. |
| Regression on RTX 4070 “proves” O(n) / O(n²) | Soften to: empirical resource scaling is **consistent with** expected complexity *trends*; hardware timings do not prove asymptotics. |

---

## 7. Statistics (flag for final-defense polish — don't blindly overclaim)

Current plan: paired t-test on per-user EER.

Keep user-level paired comparisons, but be ready to discuss:

- EER is a derived operating-point summary, not a raw observation.
- Repeated factors: users × architectures × window lengths.
- Single-seed reproducibility ≠ multi-seed robustness (already noted in README §8).

Before final defense, revisit whether a repeated-measures / rank-based
paired test better matches the score distributions. Do **not** let a lone
p-value be the only claim that “LST-Former is superior.”

---

## 8. Bottom line for your write-up

Keep:

- Clarkson II unconstrained setting
- Same HT/FT + sliding windows for all three models
- OCC autoencoders
- Security + efficiency + **window stability** as co-equal result axes

Change:

- Soften “inherently superior”
- Answer the “what does LSTM add?” question with inductive-bias + redundancy
- Make architecture × window length the intellectual center of Results
- Fix PyTorch/TF, LSTM attention wording, and complexity claim language

Then let the matrix tell the story — win, tie, or loss by condition are all
valid scientific findings.

---

## 9. What the window-stability matrix actually found

Full sweep complete (`outputs/window_stability.md`,
`window_stability_analysis.json`). Common-user set: **n=81**.

| N | LSTM EER | Transformer EER | LST-Former EER | LST-F vs Trans (p) |
|---|---:|---:|---:|---|
| 25 | 0.440 | 0.370 | **0.327** | **sig.** (p=0.003) |
| 50 | 0.451 | 0.345 | **0.311** | **sig.** (p=0.014) |
| 100 | 0.455 | 0.332 | **0.318** | not sig. (p=0.30) |

**Trajectory (paired EER N=25 → N=100):**

- LSTM: **gets worse** (+0.015, p=0.003)
- Transformer: **improves** (−0.038, p=0.001)
- LST-Former: already strong at short N; small further change not significant

**Story for the thesis (honest):**

1. Parallel fusion showed a **measurable advantage** over both single-branch
   models at short and medium windows — not only at long N.
2. As temporal context grows, the **standalone Transformer catches up**;
   by N=100 the LST-Former–Transformer gap is no longer significant.
3. That revises the “hybrid helps most when sequences are long” intuition:
   here the LSTM branch’s incremental value is clearest when the window is
   shorter; with long windows, self-attention alone largely covers what the
   hybrid was adding.
4. Efficiency still favors Transformer (fewer params, lower latency) —
   report security *and* cost together.

Suggested Results framing sentence:

> Across window lengths 25, 50, and 100, LST-Former achieved the lowest mean
> EER at every length. The advantage over Transformer was statistically
> significant at N=25 and N=50 but not at N=100, where Transformer’s
> accuracy improved with longer context and closed the gap. LSTM performance
> did not benefit from longer windows and remained clearly weaker.

**Trade-off one-liner (preferred abstract/conclusion wording):**

> LST-Former provides a measurable security advantage at shorter-to-medium
> sequence lengths, while the standalone Transformer catches up as the
> temporal window becomes longer. This advantage comes at a computational
> cost.

---

## 10. Experiment B — is the advantage broad or subset-driven?

See `outputs/delta_eer_analysis.md` (n=81 common users).

| N | Mean ΔEER | % users where LST-F better | Top-20% winners' share of +mass | sig? |
|---|---:|---:|---:|---|
| 25 | 0.043 | 61.7% | 58.2% | yes |
| 50 | 0.034 | 65.4% | 64.5% | yes |
| 100 | 0.013 | 56.8% | 65.2% | no |

**Reading:** A *majority* of users improve at N=25/50, but a large share of
the *total* positive ΔEER mass comes from the top quintile of winners
(concentrated magnitude). Frame carefully:

- Safe: “LST-Former improved EER for a majority of users at short/medium N;
  the mean advantage is statistically significant but heterogeneous in size.”
- Unsafe: “LST-Former uniformly / inherently outperforms Transformer.”

---

## 11. Open investigation — why ablation ≠ window-stability?

**Resolved at N=50 on the same 81 users** (`outputs/lstformer_ablation_aligned_N50.json`):

| Fusion | Mean EER | vs Transformer (p) |
|---|---:|---|
| concat (original) | 0.448 | **sig. worse** (p≈0) |
| local_injection | 0.341 | not sig. (p=0.74) — tied |
| hybrid_nobottleneck (re-train) | 0.325 | not sig. (p=0.16) |
| hybrid_nobottleneck (window-stability run) | 0.311 | **sig.** (p=0.014) |

**What this means:**

1. The old n≈27 ablation finding was **not** just an underpowered fluke for
   `local_injection` — on n=81 it still ties Transformer. Fusion style matters.
2. **`hybrid_nobottleneck` is the only fusion that numerically beats
   Transformer** on this matched set; `concat` is clearly worse; `local_injection`
   is essentially Transformer-equivalent.
3. Significance for hybrid vs Transformer is **seed-/run-sensitive**: the
   window-stability draw is significant (p=0.014); an independent re-train of
   the same architecture on the same users lands at p=0.16. Do **not** oversell
   a single p-value — report the trade-off trajectory across N and the fusion
   ranking, and note single-seed fragility (README §8).

**Thesis-safe claim:** The measurable short/medium-N advantage associated with
LST-Former depends on the **hybrid_nobottleneck** design (not generic
LSTM+Transformer fusion). Original concat fusion does not deliver it.

---

## 12. Complexity wording (fix in the PDF)

**Theoretical:** LSTM O(N); Transformer / LST-Former O(N²) because of the
attention branch. Established from the algorithm, not from your GPU.

**Empirical:** report log-log exponents and say measurements are
**consistent with** those trends. Do **not** write “regression proves O(N²).”
`resource_scaling.py` / `src/complexity.py` now emit log-log p and
multi-candidate fits with this claim language built in.

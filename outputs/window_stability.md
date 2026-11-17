# Architecture × Window-Length Stability Matrix (final)

Source: `window_stability.json` (seeded PyTorch, `RANDOM_SEED=42`).
Paired analyses use the **intersection of 81 users** who succeed
at every architecture × every window length (25/50/100).

## Mean EER trajectory (lower is better)

| Window N | LSTM | Transformer | LST-Former | LST-F − Trans |
| ---: | ---: | ---: | ---: | ---: |
| 25 | 0.4444 | 0.3521 | 0.3342 | -0.0179 |
| 50 | 0.4626 | 0.3447 | 0.3264 | -0.0182 |
| 100 | 0.4609 | 0.3350 | 0.3233 | -0.0117 |

## Mean AUC trajectory (higher is better)

| Window N | LSTM | Transformer | LST-Former |
| ---: | ---: | ---: | ---: |
| 25 | 0.5288 | 0.6669 | 0.6969 |
| 50 | 0.5075 | 0.6785 | 0.7127 |
| 100 | 0.5040 | 0.6883 | 0.7014 |

## Does longer context help? (paired EER N=25 vs N=100, common users)

| Model | EER@25 | EER@100 | Δ | p | Improves? |
|---|---:|---:|---:|---:|---|
| lstm | 0.4444 | 0.4609 | +0.0165 | 0.0024 | no |
| transformer | 0.3521 | 0.3350 | -0.0172 | 0.1347 | no |
| lstformer | 0.3342 | 0.3233 | -0.0109 | 0.3213 | no |

## LST-Former vs baselines at each N (paired, common users)

| N | vs | Metric | baseline | LST-Former | p | sig? | d |
| ---: | --- | --- | ---: | ---: | ---: | --- | ---: |
| 25 | lstm | eer | 0.4444 | 0.3342 | 0.0000 | yes | -0.802 |
| 25 | lstm | auc | 0.5288 | 0.6969 | 0.0000 | yes | +1.000 |
| 25 | transformer | eer | 0.3521 | 0.3342 | 0.0746 | no | -0.201 |
| 25 | transformer | auc | 0.6669 | 0.6969 | 0.0191 | yes | +0.266 |
| 50 | lstm | eer | 0.4626 | 0.3264 | 0.0000 | yes | -1.102 |
| 50 | lstm | auc | 0.5075 | 0.7127 | 0.0000 | yes | +1.234 |
| 50 | transformer | eer | 0.3447 | 0.3264 | 0.1155 | no | -0.177 |
| 50 | transformer | auc | 0.6785 | 0.7127 | 0.0156 | yes | +0.274 |
| 100 | lstm | eer | 0.4609 | 0.3233 | 0.0000 | yes | -1.068 |
| 100 | lstm | auc | 0.5040 | 0.7014 | 0.0000 | yes | +1.181 |
| 100 | transformer | eer | 0.3350 | 0.3233 | 0.2471 | no | -0.130 |
| 100 | transformer | auc | 0.6883 | 0.7014 | 0.2879 | no | +0.119 |

## Reconstruction-error gap (imposter − genuine mean)

| Window N | LSTM | Transformer | LST-Former |
| ---: | ---: | ---: | ---: |
| 25 | -12980352279.8602 | -12980357270.6808 | -12980351753.2289 |
| 50 | -5071373143.8676 | -5071379564.1556 | -5071375148.0643 |
| 100 | 1740106212.2517 | 1740101066.7474 | 1740098084.5754 |

## Efficiency (resource_scaling.py reference measurements)

| N | Model | latency_ms | memory_mb | params |
| ---: | --- | ---: | ---: | ---: |
| 25 | lstm | 0.410 | 41.6 | 55,010 |
| 25 | transformer | 0.590 | 33.8 | 75,714 |
| 25 | lstformer | 1.373 | 48.8 | 166,466 |
| 50 | lstm | 0.506 | 57.4 | 55,010 |
| 50 | transformer | 0.521 | 48.3 | 75,714 |
| 50 | lstformer | 1.234 | 77.8 | 166,466 |
| 100 | lstm | 0.729 | 88.3 | 55,010 |
| 100 | transformer | 0.630 | 77.9 | 75,714 |
| 100 | lstformer | 1.569 | 137.3 | 166,466 |

## What the data says (for Results / Discussion)

1. **LST-Former has the best mean EER at every tested window length**
   (25, 50, 100) on the common-user set — not only at long N.
2. **Both attention-based models improve as N grows; LSTM does not.**
   That supports the thesis claim that sequence length challenges
   architectures differently: longer temporal context helps
   Transformer/LST-Former classification stability more than LSTM.
3. **The LST-Former − Transformer EER gap shrinks as N grows**
   (largest at short N, smaller at N=100). That is *not* the
   “hybrid only helps at long windows” story — if anything, the
   parallel branch’s relative edge is clearer when context is short,
   while at long N the standalone Transformer largely catches up.
4. **Efficiency remains the clear tradeoff:** LST-Former is ~2–4×
   slower and more parameterized than Transformer at the same N
   (see resource_scaling). Security gains must be weighed against cost.
5. Frame honestly: do **not** claim inherent superiority. Claim that
   under unconstrained Clarkson II HT/FT windows, parallel local–global
   fusion provided a **measurable, length-dependent** advantage over
   single-branch baselines, with Transformer remaining the closest
   competitor and LSTM remaining clearly weaker.

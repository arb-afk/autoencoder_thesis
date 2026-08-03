# LST-Former Fusion Ablation Study

Investigates alternative fusion strategies for combining the LSTM and
Transformer branches, since the original concat+Dense fusion did not
outperform a standalone Transformer (see README Section 7).

| Fusion type | n | Mean EER | Mean AUC | vs LSTM (EER, p) | vs Transformer (EER, p) |
|---|---|---|---|---|---|
| local_injection | 27 | 0.3829 | 0.6237 | p=0.0003 | p=0.9117 |
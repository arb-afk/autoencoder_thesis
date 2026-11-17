# LST-Former Fusion Ablation Study

Investigates alternative fusion strategies for combining the LSTM and
Transformer branches, since the original concat+Dense fusion did not
outperform a standalone Transformer (see README Section 7).

| Fusion type | n | Mean EER | Mean AUC | vs LSTM (EER, p) | vs Transformer (EER, p) |
|---|---|---|---|---|---|
| concat | 81 | 0.4481 | 0.5165 | p=0.3320 | p=0.0000 |
| local_injection | 81 | 0.3406 | 0.6874 | p=0.0000 | p=0.7431 |
| hybrid_nobottleneck | 81 | 0.3251 | 0.7121 | p=0.0000 | p=0.1609 |
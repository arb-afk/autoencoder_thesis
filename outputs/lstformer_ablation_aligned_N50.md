# LST-Former Fusion Ablation Study (leak-free)

N=50, n=81, train-only z-score / per-partition windows.

| Fusion type | n | Mean EER | Mean AUC | vs LSTM (EER, p) | vs Transformer (EER, p) |
|---|---|---|---|---|---|
| concat | 81 | 0.4581 | 0.5113 | p=0.0827 | p=0.0000 |
| local_injection | 81 | 0.3547 | 0.6553 | p=0.0000 | p=0.3590 |
| hybrid_nobottleneck | 81 | 0.3228 | 0.7033 | p=0.0000 | p=0.0516 |

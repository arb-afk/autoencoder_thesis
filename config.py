"""Hyperparameters and paths aligned with thesis methodology (Chapter 3)."""

from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent
# Clarkson II filtered logs — NOT shipped in git. Place/extract the dataset
# so this directory exists, or change DATA_DIR to your local path.
# Expected layout: .../filtered_dataset/<per-user keystroke files>
DATA_DIR = ROOT_DIR / "clarkson-II-2018-filtered_dataset (1)" / "filtered_dataset"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"

# Feature extraction
FLIGHT_TIME_CLIP_MS = 2000  # temporal clipping threshold (Section 3.3.2)
FEATURE_DIM = 2  # Hold Time + Flight Time

# Sliding windows (Section 3.3.3)
WINDOW_SIZE = 50  # N keystrokes per window
STRIDE = 25  # 50% overlap

# Data split (Section 3.5.1) — chronological
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# Training (Section 3.5.2–3.5.3)
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
LOSS_FN = "mse"

# Architecture-specific tuning (Section 3.5.2 revision -- see README Section
# 7.6 for the full story). Tried: a linear LR warm-up over the attention
# models' first WARMUP_STEPS steps (standard practice for self-attention,
# e.g. Vaswani et al. 2017), on the theory that a flat LR isn't a fair
# "equal playing field" across architectures with very different loss
# landscapes. Empirically tested on the full 88-user run and found to make
# things WORSE for both Transformer and LST-Former, not better -- with
# early stopping and small per-user datasets (median ~32 epochs), the
# warmup ate a large fraction of many users' entire training budget, so
# most users never trained at full LR before stopping. USE_LR_WARMUP is
# therefore OFF by default; the option is kept (rather than deleted) so
# this negative result is reproducible. LSTM's gradient-norm clipping had
# no such downside (negligible/slightly positive effect) and stays enabled.
USE_LR_WARMUP = False
WARMUP_STEPS = 100
LSTM_GRAD_CLIP_NORM = 1.0

# Reproducibility: without a fixed seed, re-running main.py with identical
# config gives different results each time purely from random weight init /
# data shuffling / imposter sampling -- e.g. LSTM's EER drifted by ~0.004
# between two runs with byte-identical config during the warmup
# investigation (README Section 7.6). That's a real gap for a thesis whose
# numbers should be reproducible. main.py seeds Python/NumPy/PyTorch (and
# enables deterministic cuDNN) with this value at startup.
RANDOM_SEED = 42

# Model architecture (defaults when not specified in thesis)
HIDDEN_DIM = 64
LATENT_DIM = 32
D_MODEL = 64
NHEAD = 4
NUM_TRANSFORMER_LAYERS = 2
DROPOUT = 0.1

# LST-Former fusion strategy (Section 3.4.3). "hybrid_nobottleneck" is the
# validated design -- see outputs/lstformer_ablation.md and README Section 7
# for why the original "concat" fusion underperformed and how this was found.
# Other options ("concat", "norm_concat", "gated", "weighted_sum") are kept
# in src/models/lstformer_ae.py for reproducing that ablation study.
LSTFORMER_FUSION_TYPE = "hybrid_nobottleneck"

# Evaluation / scaling sweep lengths
# Used by resource_scaling.py and as the default window-stability set.
# Prefer a denser grid when re-running complexity analysis so log-log
# empirical exponents are better constrained.
SEQUENCE_LENGTHS = [25, 50, 100]
EXTENDED_SEQUENCE_LENGTHS = [25, 50, 75, 100, 150, 200, 300, 400, 600, 800]

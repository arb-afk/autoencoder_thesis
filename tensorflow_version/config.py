"""
Hyperparameters and paths for the TensorFlow/Keras port (Section 3.7.2).
Mirrors ../config.py exactly so results are comparable across frameworks.
"""

from pathlib import Path

# Paths (dataset lives at the repo root, shared by both framework versions).
# Dataset is NOT in git — see root README §2.0.
ROOT_DIR = Path(__file__).parent
REPO_ROOT = ROOT_DIR.parent
DATA_DIR = REPO_ROOT / "clarkson-II-2018-filtered_dataset (1)" / "filtered_dataset"
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

# Architecture-specific tuning (Section 3.5.2 revision) -- see the root
# config.py's comment and README Section 7.6 for the rationale, including
# why USE_LR_WARMUP is OFF by default (empirically hurt accuracy). Mirrors
# the PyTorch version exactly.
USE_LR_WARMUP = False
WARMUP_STEPS = 100
LSTM_GRAD_CLIP_NORM = 1.0

# Reproducibility -- see the root config.py's comment and README Section 7.6.
RANDOM_SEED = 42

# Model architecture (defaults when not specified in thesis)
HIDDEN_DIM = 64
LATENT_DIM = 32
D_MODEL = 64
NHEAD = 4
NUM_TRANSFORMER_LAYERS = 2
DROPOUT = 0.1

# LST-Former fusion strategy (Section 3.4.3) -- see the root config.py's
# comment and README Section 7 / outputs/lstformer_ablation.md for why this
# is "hybrid_nobottleneck" rather than the original "concat" fusion.
LSTFORMER_FUSION_TYPE = "hybrid_nobottleneck"

# Evaluation
SEQUENCE_LENGTHS = [25, 50, 100]  # resource-scaling analysis (Section 3.9.2)

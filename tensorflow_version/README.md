# TensorFlow / Keras port

Parallel implementation of the PyTorch thesis pipeline (TensorFlow 2.16.1 +
Keras), matching Section 3.7.2.

**Use the [root README](../README.md) for:** clone/setup, Clarkson II data
placement, canonical n=81 protocol, and how to interpret results.

## Quick start

```powershell
# Python 3.12 required
cd tensorflow_version
python -m venv .venv312
.\.venv312\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --model all --max-users 20
```

On Windows, native TF ≥2.11 has **no GPU**. For CUDA, use WSL2 — see root
README §2.7 (`run_wsl.sh`, `check_gpu.sh`).

Dataset path is shared with the repo root via `config.DATA_DIR` (points at
`../clarkson-II-2018-filtered_dataset (1)/filtered_dataset`).

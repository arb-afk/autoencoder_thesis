# Keystroke Anomaly Detection — LSTM vs. Transformer vs. LST-Former

Implementation of the three autoencoder architectures compared in the thesis
*"Performance and Computational Efficiency Analysis of LSTM, Transformer, and
Parallel LST-Former Autoencoders for Continuous Keystroke Anomaly Detection"*,
trained and evaluated on the **Clarkson II Keystroke Dataset**.

**Primary thesis results (locked):** architecture × window-length matrix on a
canonical **n=81** common-user set. See
[`outputs/CANONICAL_PROTOCOL.md`](outputs/CANONICAL_PROTOCOL.md) before citing
numbers — do not mix n=88 / n≈27 exploratory runs into primary tables.

There are **two parallel implementations**:

| | Framework | Python | Location | GPU support |
|---|---|---|---|---|
| **Main version** | PyTorch | 3.10+ (3.14 OK) | repo root | Yes (CUDA on Windows/Linux) |
| **Thesis-exact version** | TensorFlow 2.16.1 + Keras | **3.12 only** | `tensorflow_version/` | Not native on Windows; use **WSL2** — see [§2.5](#25-tensorflow-version-and-why-it-has-no-gpu-on-windows) / [§2.7](#27-wsl2-gpu-accelerated-tensorflow-already-set-up) |

---

## 0. Clone from GitHub (quick start)

```powershell
git clone https://github.com/YOUR_USER/YOUR_REPO.git
cd YOUR_REPO
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate           # Linux / macOS / WSL
pip install -r requirements.txt
```

**You still need the Clarkson II data locally** — it is **not** in this
repository (size + licensing). Place it so this path exists:

```text
<repo>/clarkson-II-2018-filtered_dataset (1)/filtered_dataset/<user_id files>
```

That exact folder name is what `config.py` → `DATA_DIR` expects. If your
extract uses a different name, either rename it or edit `DATA_DIR` in
`config.py` and `tensorflow_version/config.py`.

Then, to reproduce the primary PyTorch validation (GPU recommended):

```powershell
python main.py --model all --max-users 96 --device cuda
python window_stability.py --max-users 96 --lengths 25 50 100 --device cuda
python analyze_window_stability.py
python investigate_delta_eer.py
python statistical_analysis.py --metric eer
python resource_scaling.py --lengths 25 50 75 100 150 200 300 400 600 800 --device cuda
```

Pre-computed result tables used in the thesis write-up already live under
`outputs/` (JSON/Markdown/plots). Model weight files (`.pt`) are gitignored
and must be regenerated if you need checkpoints.

---

## 1. What's in this repo

```
.
├── config.py                  # Hyperparameters + DATA_DIR / paths
├── main.py                    # Train + evaluate 3 models (N=50 default)
├── window_stability.py        # Architecture × window centerpiece (N=25/50/100)
├── analyze_window_stability.py
├── investigate_delta_eer.py   # Per-user Delta EER (heterogeneity)
├── ablate_lstformer.py        # Fusion ablation (aligned n=81 protocol)
├── resource_scaling.py        # Efficiency vs N + log-log exponents
├── statistical_analysis.py    # Paired t-test + Cohen's d
├── requirements.txt
├── .gitignore
├── src/                       # PyTorch implementation
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── complexity.py
│   └── models/                # lstm_ae, transformer_ae, lstformer_ae
├── tensorflow_version/        # Keras/TF 2.16.1 port (optional)
├── outputs/                   # Thesis-ready results (no .pt weights)
│   ├── CANONICAL_PROTOCOL.md  # LOCKED: n=81 primary population + wording
│   ├── thesis_framing_notes.md
│   ├── window_stability*.json/.md
│   ├── delta_eer_analysis.*
│   ├── lstformer_ablation_aligned_N50.*
│   ├── results.json           # Supplementary n=88 @ N=50
│   ├── resource_scaling.json / asymptotic_report.md / plots/
│   └── common_users_81.txt
└── (local only, gitignored)
    ├── clarkson-II-2018-filtered_dataset (1)/
    ├── .venv / .venv312/
    └── outputs/models*/       # regenerated checkpoints
```

---

## 2. Setup

### 2.0 Dataset (required, not shipped)

1. Obtain the **Clarkson II** filtered keystroke dataset through your
   institution / dataset authors (same corpus used in the thesis).
2. Extract so per-user log files sit under:

   `clarkson-II-2018-filtered_dataset (1)/filtered_dataset/`

3. Confirm `config.DATA_DIR` points at that folder.

This study evaluates **unconstrained keyboard behavior**, not
activity-labeled typing. The dataset does not identify whether events came
from text entry, gaming, etc. — that is a deliberate limitation of the
experimental setting (see `outputs/thesis_framing_notes.md`).

### 2.1 Requirements

- **PyTorch path**: Python 3.10+ (3.14 works with current PyTorch wheels).
- **TensorFlow path**: Python **3.12** only for TF 2.16.1 — see §2.6–2.7.
- NVIDIA GPU + CUDA optional but recommended for full sweeps.

### 2.2 Install dependencies

```powershell
# From repo root (after creating/activating a venv — see §0)
pip install -r requirements.txt
```

For TensorFlow: `pip install -r tensorflow_version/requirements.txt` inside a
Python 3.12 environment (or the WSL2 venv in §2.7).

### 2.3 CPU vs. GPU (Section 3.7.1)

Training may use GPU; **inference latency is always measured on CPU**
(`measure_inference_latency`), matching Section 3.7.1. Peak VRAM is logged
when CUDA is available. Without a GPU, everything falls back to CPU
automatically.

### 2.4 Why is there a PyTorch version at all, if the thesis specifies TensorFlow?

The thesis (Section 3.7.2) specifies **TensorFlow v2.16.1 + Keras**. The
PyTorch version at the repo root was built first, for one concrete reason:

> Default Python on many machines is newer than TensorFlow’s published
> wheels (e.g. 3.14). PyTorch already supports those versions, so it was
> used for primary GPU runs. The TensorFlow port lives in
> `tensorflow_version/` for framework parity / cross-check.

The **`tensorflow_version/` folder now contains the literal TensorFlow/Keras
port** (Section 2.6 below has full setup/run instructions), built by
installing Python 3.12 side-by-side with 3.14 specifically for TensorFlow
compatibility. Both versions share the same `config.py` hyperparameters and
produce `results.json` in the same schema, so `statistical_analysis.py`
works against either one's output.

### 2.5 TensorFlow version and why it has no GPU on Windows

Unlike the PyTorch version, **native Windows TensorFlow trains and infers on
CPU only** when using official `pip install tensorflow` wheels for 2.16.x.
This is not a project bug — it's a TensorFlow/Windows platform limitation:

> Since **TensorFlow 2.11**, Google stopped shipping GPU (CUDA) support in
> the native Windows `pip install tensorflow` package. The Windows wheel is
> CPU-only from 2.11 onward, including 2.16.1 (the version Section 3.7.2
> specifies). Genuine GPU-accelerated TensorFlow on Windows now requires
> either:
> 1. **WSL2** (Windows Subsystem for Linux) — install a Linux distro, then
>    `pip install tensorflow[and-cuda]` inside it. This gives you real CUDA
>    acceleration, but it's a bigger, separate setup (a Linux environment
>    running alongside Windows).
> 2. **`tensorflow-directml-plugin`** — an unofficial Microsoft-maintained
>    plugin that accelerates TensorFlow via DirectX 12 instead of CUDA. It
>    works natively on Windows with any DirectX12 GPU, but it's a
>    third-party plugin (not what Section 3.7.2 literally specifies), lags
>    behind current TensorFlow versions, and has had spotty maintenance.

Practically, this means: **native Windows** TensorFlow numbers in this repo
are a CPU-only baseline. The architectures, metrics (FAR/FRR/EER/AUC), and
Big-O scaling trends (Section 3.10.2) are all still valid and comparable to
the PyTorch CPU numbers. **WSL2 has since been set up** (Section 2.7) to get
genuine GPU-accelerated TensorFlow numbers too, for a full three-way
cross-check (PyTorch GPU / TensorFlow CPU / TensorFlow GPU) — see Section 7.5.

### 2.6 Setting up and running the TensorFlow version

Use a **Python 3.12** virtual environment (TF 2.16.1 has no wheels for 3.13+).
Local `.venv312` folders are gitignored — create your own:

```powershell
cd tensorflow_version
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --model all --max-users 5
python resource_scaling.py
```

Ensure the Clarkson II folder exists at the repo-root path from §2.0
(`tensorflow_version/config.py` points at `../clarkson-II-.../filtered_dataset`).

Or without activating, call the interpreter directly:

```powershell
cd D:\School\Thesis\tensorflow_version
..\.venv312\Scripts\python.exe main.py --model lstm --user 118630
..\.venv312\Scripts\python.exe resource_scaling.py --lengths 25 50 100
```

All the same CLI flags from Section 3 below work identically
(`--model`, `--user`, `--max-users`, `--lengths`); the only difference is
`--device cuda` has no effect on Windows (see Section 2.5) — it always runs
on CPU. Statistical analysis can be run from the root against the TF
results:

```powershell
cd D:\School\Thesis
python statistical_analysis.py --results tensorflow_version\outputs\results.json
```

To set up the environment from scratch elsewhere:

```powershell
winget install --id Python.Python.3.12 -e
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r tensorflow_version\requirements.txt
```

**A subtlety worth knowing if you inspect `tensorflow_version/src/evaluate.py`:**
calling a Keras model directly in eager mode (`model(x)`) inside a tight
timing loop has large per-call Python/dispatch overhead — an early test
measured ~139ms per inference, ~300x slower than the PyTorch version for the
*same architecture*, which would have been a misleading comparison. Wrapping
the forward pass in `@tf.function` (compiling it once during warmup, then
reusing the compiled graph) fixed this, bringing latency down to a
comparable single-digit-millisecond range. This is standard practice for
benchmarking TensorFlow models and mirrors how they'd actually be deployed.

### 2.7 WSL2 GPU-accelerated TensorFlow (already set up)

To get genuine CUDA-accelerated TensorFlow numbers (Section 2.5's option 1),
WSL2 + Ubuntu has been installed on this machine with a working GPU-passthrough
TensorFlow setup. If you need to redo this on another machine, here's exactly
what was done and the gotchas that came up:

1. **Install WSL2 + Ubuntu** (needs Administrator once): `wsl --install`,
   reboot, then `wsl` to finish first-run Linux user setup.
2. **GPU passthrough is automatic** on modern NVIDIA drivers — verify with
   `wsl nvidia-smi` (no separate Linux NVIDIA driver install needed; WSL2
   uses the Windows host driver).
3. **Install Python 3.12 inside Ubuntu** (its default Python was too new for
   TensorFlow, same issue as Section 2.4): `sudo apt install
   software-properties-common && sudo add-apt-repository ppa:deadsnakes/ppa
   && sudo apt install python3.12 python3.12-venv`.
4. **Create the venv on the native Linux filesystem, NOT under `/mnt/c` or
   `/mnt/d`.** This is the one gotcha that will silently break things:
   `python3.12 -m venv` on a Windows-mounted drive (drvfs) produces a broken
   venv missing `bin/activate` and other scripts, with no clear error. Create
   it in the Linux home directory instead (`python3.12 -m venv
   ~/tf_wsl_venv`), then just `cd /mnt/d/...` to run your actual project code
   — the venv and the code don't need to be on the same filesystem.
5. **Install `tensorflow[and-cuda]==2.16.1`** — this pulls the TensorFlow
   wheel plus all the `nvidia-cu12-*` CUDA/cuDNN pip packages it needs (~2GB).
6. **Fix `LD_LIBRARY_PATH`** — even after step 5, TensorFlow won't find the
   CUDA libraries dlopen needs unless `LD_LIBRARY_PATH` includes each
   `nvidia/*/lib` directory under the venv's site-packages. This has been
   patched permanently into `~/tf_wsl_venv/bin/activate` (auto-runs on every
   `source activate`), so you don't need to redo it.
7. **Pin `numpy==1.26.4`, `scipy==1.17.1`, `scikit-learn==1.9.0` explicitly.**
   TensorFlow 2.16.1 requires `numpy<2.0`, but installing `scikit-learn`/`scipy`
   without version pins pulls the latest versions, which require `numpy>=2.0`
   — a silent conflict that breaks at import time (`AttributeError: module
   'numpy' has no attribute 'long'`), not at install time. These exact
   versions are confirmed compatible together.

**Running things:** `tensorflow_version/run_wsl.sh` is a wrapper that
activates the venv and runs a script against the project code on `/mnt/d`:

```bash
wsl bash tensorflow_version/run_wsl.sh main.py --model all --max-users 20 --output outputs/results_gpu.json
wsl bash tensorflow_version/run_wsl.sh resource_scaling.py --lengths 25 50 100 200 --device cuda
```

(from PowerShell at the repo root; or drop the `wsl bash` prefix if you're
already inside a WSL Ubuntu shell). `tensorflow_version/check_gpu.py` /
`check_gpu.sh` are small standalone scripts to re-verify GPU visibility if
something seems off (`bash tensorflow_version/check_gpu.sh`).

The `--output` flag on `main.py` (TensorFlow version only) lets you point
results at a separate file (e.g. `results_gpu.json`) instead of the default
`outputs/results.json`, so a GPU run doesn't overwrite an existing CPU run's
data (or vice versa) — see Section 7.5 for why this matters here.

---

## 3. Running the pipeline

*(This section covers the **PyTorch version** at the repo root. For the
TensorFlow version, see Section 2.6 — the same flags apply, just run from
`tensorflow_version/` with `..\.venv312\Scripts\python.exe`.)*

### 3.1 Train and evaluate the models

```powershell
# Train all 3 architectures on the first 5 users found in the dataset
python main.py --model all

# Train just one architecture
python main.py --model lstm
python main.py --model transformer
python main.py --model lstformer

# Train/evaluate a specific user
python main.py --model lstformer --user 118630

# Run across more (or all 96) users — this takes longer
python main.py --model all --max-users 96

# Force a specific device
python main.py --model all --device cpu
python main.py --model all --device cuda
```

Each run:
1. Loads every user's raw keystroke log from `clarkson-II-2018-filtered_dataset (1)/filtered_dataset/`
2. Extracts Hold Time / Flight Time, Z-score normalizes per user, builds sliding windows
3. Splits chronologically 70% train / 10% validation / 20% test
4. Trains the model **only on that user's genuine data** (One-Class Classification)
5. Tests it against its own held-out genuine data + a sample of other users' windows (zero-effort imposters)
6. Computes FAR, FRR, EER, AUC, CPU inference latency, parameter count
7. Saves trained weights to `outputs/models/<model_name>/<user_id>.pt`
8. Saves all metrics to `outputs/results.json`

**Note:** models are trained per-user, so `--max-users 96` trains and
evaluates 96 separate models per architecture. Expect a long run — start
with `--max-users 5` or so to sanity-check things first.

### 3.2 Resource-scaling analysis (Section 3.9.2)

Measures how parameter count, training memory, and inference latency scale
with sequence length N, and fits the theoretically-expected curve (linear
for LSTM, quadratic for Transformer/LST-Former) to verify Big-O growth
empirically (see Section 5 below for more on this).

```powershell
python resource_scaling.py
python resource_scaling.py --lengths 25 50 100 200 400   # test more/larger N
python resource_scaling.py --user 118630                 # pick a specific user's data

# Architecture x window-length centerpiece (security + recon + efficiency matrix)
python window_stability.py --max-users 96 --lengths 25 50 100 --device cuda
# Writes outputs/window_stability.json and outputs/window_stability.md
# Resume-safe: merges into the JSON after each model x N cell completes.
```

Output: `outputs/resource_scaling.json` + a printed regression summary with
R² values.

**Caveat (PyTorch version):** this machine has a CUDA GPU, so `--device cuda`
gives a true VRAM measurement via `torch.cuda.max_memory_allocated()`. On a
machine without an NVIDIA GPU, "peak memory" falls back to a host-RAM proxy
via `psutil` rather than true VRAM, and the memory R² is noisier — treat it
as indicative only; the latency and parameter-count R² values are reliable
either way.

**Caveat (TensorFlow version):** since the TensorFlow port has no GPU
available on Windows (Section 2.5), it *always* uses the `psutil` host-RAM
proxy for memory, regardless of `--device`. Its latency and parameter-count
R² values are still reliable evidence for the Big-O claims (a local run
matched the PyTorch version's conclusions: linear fit for LSTM, quadratic
for Transformer/LST-Former, R² ≈ 0.997–1.000 for latency).

### 3.3 Statistical analysis (Section 3.10.1)

Runs a paired t-test + Cohen's d between the LST-Former and each baseline,
on a per-user metric (default EER). Requires `outputs/results.json` to
already contain results for **all three models on the same users** — i.e.
run `main.py --model all` first.

```powershell
python main.py --model all --max-users 96
python statistical_analysis.py
python statistical_analysis.py --metric auc     # or far, frr
python statistical_analysis.py --results tensorflow_version\outputs\results.json --metric eer
```

Output: `outputs/statistical_analysis_<metric>.json` (one file per metric,
so `--metric eer` and `--metric auc` don't overwrite each other) + a
printed report with t-statistic, p-value, significance at alpha=0.05, and
Cohen's d with a plain-language effect size label
(negligible/small/medium/large). Use `--output <path>` to override the
save location.

---

## 4. What to change, and what it affects

All tunable parameters live in **`config.py`**. Here's what each one does
and when you'd want to change it:

| Parameter | Default | What changing it does |
|---|---|---|
| `FLIGHT_TIME_CLIP_MS` | `2000` | Caps outlier pauses (e.g. user stops to think) before normalization. Lower it to make the model more sensitive to short pauses; raise it if genuine users have long natural pauses being clipped too aggressively. |
| `WINDOW_SIZE` | `50` | Number of consecutive keystrokes per sample fed to the model. **Bigger window** = more context per decision (usually better accuracy) but slower inference and more memory, especially for the Transformer/LST-Former (`O(N^2)`). **Smaller window** = faster, more responsive, but noisier decisions. |
| `STRIDE` | `25` | Step size between windows (50% overlap at default). **Smaller stride** = more training windows (slower to train, less risk of missing anomalies between windows) but more redundant/correlated data. **Larger stride** (up to `WINDOW_SIZE`, i.e. no overlap) = fewer, more independent windows, faster preprocessing, but less data. |
| `TRAIN_RATIO` / `VAL_RATIO` / `TEST_RATIO` | `0.70` / `0.10` / `0.20` | Chronological split proportions. Must sum to 1.0. More training data usually helps the model learn the user's baseline better; more test data gives a more stable EER/AUC estimate but with less training data per user. |
| `LEARNING_RATE` | `0.001` | Adam step size (the *post-warmup* / steady-state LR for every model — see `WARMUP_STEPS` below). Lower it (e.g. `0.0005`) if training loss is unstable/diverging; raise it (e.g. `0.003`) if training is converging too slowly and you have epochs to spare. |
| `BATCH_SIZE` | `64` | Number of windows per training step. Larger batches = smoother gradients, more RAM/VRAM use, fewer steps per epoch. Smaller batches = noisier but sometimes better-generalizing training, slower per epoch. **Must have at least this many windows in a user's training set**, or that user is skipped. |
| `MAX_EPOCHS` | `100` | Hard cap on training length. Raise this if models are still improving when early stopping isn't triggered yet (rare); lower it to speed up experimentation at the cost of potentially under-trained models. |
| `EARLY_STOPPING_PATIENCE` | `10` | Epochs to wait without validation-loss improvement before stopping. Lower it (e.g. `5`) for faster experiments; raise it (e.g. `20`) if validation loss is noisy and stops too early. |
| `USE_LR_WARMUP` | `False` | **Transformer and LST-Former only.** Whether to apply the LR warm-up described by `WARMUP_STEPS` below. Off by default — tested and found to show no reproducible benefit (and possibly a downside) in this dataset's small-per-user, early-stopped training regime; see Section 7.6(b) for the full investigation before turning this on. |
| `WARMUP_STEPS` | `100` | Only has an effect if `USE_LR_WARMUP=True`. Number of training *steps* (batches, not epochs) over which the LR linearly ramps from 0 up to `LEARNING_RATE`, before holding steady — standard practice for attention layers, which are sensitive to large early gradients. If re-enabling this, consider a smaller value (e.g. `20`) so it doesn't eat a large share of a small user's short training run (Section 7.6(b)). |
| `LSTM_GRAD_CLIP_NORM` | `1.0` | **LSTM only, always applied.** Caps the gradient norm at this value each step, guarding against the occasional large gradient spike vanilla LSTMs are prone to. Lower it (e.g. `0.5`) if LSTM training loss still shows sharp spikes; raise it or set very high (e.g. `10.0`) to effectively disable clipping. |
| `RANDOM_SEED` | `42` | Seeds Python/NumPy/PyTorch (or TensorFlow) and enables deterministic cuDNN at the start of every architecture's run, so re-running `main.py`/`ablate_lstformer.py` with identical config reproduces identical EER/AUC (see Section 7.6(b) — without this, results drifted noticeably run-to-run). Change it to check how sensitive results are to the specific seed, e.g. as a cheap proxy for a full multi-seed study. |
| `HIDDEN_DIM` | `64` | LSTM hidden state size (used in the baseline LSTM and the LSTM branch of LST-Former). Bigger = more capacity to model complex typing rhythms, but more parameters/compute. |
| `LATENT_DIM` | `32` | Size of the compressed bottleneck vector. Smaller = more aggressive compression (may lose detail, but stronger anomaly signal if the user's behavior is genuinely simple); bigger = more detail retained (may reconstruct imposters too well, hurting detection). |
| `D_MODEL` | `64` | Transformer embedding dimension (used in the Transformer and the attention branch of LST-Former). Must be divisible by `NHEAD`. |
| `NHEAD` | `4` | Number of self-attention heads. More heads let the model attend to different aspects of typing rhythm in parallel; must evenly divide `D_MODEL`. |
| `NUM_TRANSFORMER_LAYERS` | `2` | Depth of the Transformer encoder stack. More layers = more capacity and quadratic-in-N compute per layer, but risk of overfitting on the relatively small per-user datasets here. |
| `DROPOUT` | `0.1` | Regularization strength. Raise it (e.g. `0.2`–`0.3`) if models overfit quickly (train loss keeps dropping, val loss plateaus/rises early); lower it if models are underfitting. |
| `SEQUENCE_LENGTHS` | `[25, 50, 100]` | The set of window lengths swept by `resource_scaling.py` to verify Big-O complexity. Add more/larger values (e.g. `[25, 50, 100, 200, 400]`) for a more convincing regression fit, at the cost of a longer scaling run. |

**Command-line flags** (don't require editing `config.py`):

| Flag | Script | Effect |
|---|---|---|
| `--model {lstm,transformer,lstformer,all}` | `main.py` | Which architecture(s) to run |
| `--user <id>` | `main.py`, `resource_scaling.py` | Restrict to one user (filename under `filtered_dataset/`) |
| `--max-users <N>` | `main.py` | How many users to loop over when `--user` isn't given |
| `--device {cpu,cuda}` | `main.py`, `resource_scaling.py` | Force CPU or GPU |
| `--lengths <N N N...>` | `resource_scaling.py` | Override `SEQUENCE_LENGTHS` for this run |
| `--metric {eer,auc,far,frr}` | `statistical_analysis.py` | Which per-user metric to compare statistically |

---

## 5. Finding the Asymptotic (Big-O) Notation of each model

There are two complementary ways to get this, and this repo gives you both
(commands below are for the PyTorch version; for TensorFlow, `cd
tensorflow_version` and use `..\.venv312\Scripts\python.exe` instead of
`python` — everything else is identical):

### 5.1 Theoretical (read it off the architecture)

Run:

```powershell
python -c "from src.complexity import print_theoretical_summary; print_theoretical_summary()"
```

This prints the reasoning for each model:

- **LSTM — O(N) time & space.** It processes the window one keystroke at a
  time through a recurrent cell; each of the N steps is constant-size work.
- **Transformer — O(N²) time & space.** Self-attention computes an N×N
  score matrix (every keystroke attends to every other keystroke).
- **LST-Former — O(N²) overall** (`O(N) + O(N²)`, dominated by its
  Transformer branch), though with a smaller constant factor than a
  standalone deep Transformer since the LSTM branch offloads local-rhythm
  modeling.

### 5.2 Empirical (measure it and fit a curve — what the thesis asks for in Section 3.10.2)

```powershell
python resource_scaling.py
```

This measures latency (and parameter count / memory) at N = 25, 50, 100,
then fits:
- a **straight line** (`y = a·N + b`) to the LSTM's numbers, and
- a **parabola** (`y = a·N² + b·N + c`) to the Transformer's and
  LST-Former's numbers,

reporting the **R² (coefficient of determination)** for each fit. An R²
close to 1.0 means that order of growth explains the measured data well —
this is the direct empirical evidence for the Big-O claims. In a local test
run, latency fits came back at R² ≈ 0.997 (LSTM, linear) and R² ≈ 1.000
(Transformer/LST-Former, quadratic), strongly confirming the theoretical
complexity.

For a more convincing/publishable curve, widen the sweep, e.g.:

```powershell
python resource_scaling.py --lengths 25 50 100 200 400 800
```

(Larger N will take longer to run and need a user file with enough raw
keystrokes — the script automatically searches for one with enough data;
use `--user <id>` to point at a specific large file yourself.)

### 5.3 "Recording" the result for your thesis document

Every run of `resource_scaling.py` automatically writes, under `outputs/`:

- **`resource_scaling.json`** — raw numbers + fitted coefficients/R², for archival/re-analysis
- **`plots/scaling_latency.png`**, **`scaling_params.png`**, **`scaling_memory.png`** — one figure per metric, with each model's measured points and its fitted curve overlaid (ready to drop straight into your Results chapter as a figure)
- **`asymptotic_report.md`** — a self-contained Markdown write-up: the theoretical complexity table, the fitted-equation + R² table per model, the figures, and the raw measurement tables. Open it directly, or copy sections of it straight into your thesis document.

This is the most direct answer to "how do I record the asymptotic
notation": run `python resource_scaling.py`, then open
`outputs/asymptotic_report.md` (and the PNGs alongside it) — that file
already has the theoretical Big-O statement, the empirical curve-fit
equations, R² values, and figures assembled in one place.

---

## 6. Interpreting the output files

All under `outputs/` for the PyTorch version, or `tensorflow_version/outputs/`
for the TensorFlow version:

| File | Produced by | Contents |
|---|---|---|
| `models/<model>/<user_id>.pt` (PyTorch) / `models/<model>/<user_id>.weights.h5` (TF) | `main.py` | Trained weights + best validation loss for that user/model |
| `results.json` | `main.py` | Per-user and averaged FAR/FRR/EER/AUC/latency/param-count for each model |
| `resource_scaling.json` | `resource_scaling.py` | Raw measurements + regression fits (coefficients, R²) per model, per sequence length |
| `plots/scaling_*.png` | `resource_scaling.py` | Measured points + fitted curve per metric (latency/params/memory) |
| `asymptotic_report.md` | `resource_scaling.py` | Self-contained Markdown write-up combining theory, fits, figures, and raw tables |
| `statistical_analysis_<metric>.json` | `statistical_analysis.py` | Paired t-test + Cohen's d results, LST-Former vs. each baseline, one file per `--metric` |

Both versions produce `results.json` in the same schema, so
`statistical_analysis.py` (framework-agnostic — pure NumPy/SciPy) works
against either one by pointing `--results` at the file you want:

```powershell
python statistical_analysis.py --results outputs\results.json                      # PyTorch
python statistical_analysis.py --results tensorflow_version\outputs\results.json   # TensorFlow
```

---

## 7. The LST-Former fusion investigation (important — read this)

**tl;dr:** the first full run showed the proposed LST-Former losing to a
plain Transformer. We investigated why, found the root cause (a design
issue in the fusion step, not a bug), fixed it, and the fixed version is
now the default in `config.py` (`LSTFORMER_FUSION_TYPE = "hybrid_nobottleneck"`).
This section documents the whole investigation so you can describe it
honestly in your Methodology/Results/Discussion chapters — it's a
legitimate, evidence-based design iteration, not post-hoc tuning.

### 7.1 The original finding

A full run (PyTorch, 88 users; TensorFlow cross-check, 18 users) with the
thesis's originally-described fusion (concat two branch latents → Dense →
shared LSTM decoder) showed the standalone **Transformer beating LST-Former**
(paired t-test, p<0.0001, medium effect size), while **LST-Former was only
statistically tied with the plain LSTM baseline** (p=0.377) — i.e. the
proposed hybrid wasn't better than either baseline. This reproduced
independently in both frameworks, so it wasn't noise or a framework bug.

### 7.2 The investigation

Run yourself with:

```powershell
python ablate_lstformer.py --max-users 30 --device cuda
```

Five architecture variants were tested on the exact same fixed 27-user
subset (fair, paired comparisons), holding everything else constant:

| Variant | Mean EER | vs. LSTM (p) | vs. Transformer (p) |
|---|---|---|---|
| concat (original) | 0.4839 | 0.0833 | 0.0001 |
| norm_concat | 0.4835 | 0.2044 | 0.0001 |
| gated | 0.4837 | 0.2271 | 0.0001 |
| weighted_sum | 0.4893 | 0.8290 | 0.0000 |
| concat + Transformer-style decoder | 0.4901 | 0.9807 | 0.0000 |
| **hybrid_nobottleneck** | **0.3724** | **0.0001** | **0.8894** |

**Four different fusion strategies, and even swapping the decoder type,
all landed in the same tight cluster** — none closed the gap. That
consistent null result ruled out "wrong fusion formula" as the explanation
and pointed at something structural.

**Root cause:** the Transformer-AE baseline never compresses the window to
a single vector — self-attention keeps every timestep's full contextual
information throughout, so it reconstructs from strictly more information
than a real bottleneck autoencoder. LST-Former (and the LSTM-AE), by
design, compress the whole window into **one fixed-size latent vector**
before decoding — a genuine information bottleneck. The problem: LST-Former's
Transformer *branch* was also being pooled into that same single-vector
bottleneck before fusion, so it never got to use the exact property that
makes the standalone Transformer strong. No amount of changing the fusion
formula or decoder afterward could recover information that was already
discarded at the pooling step.

### 7.3 The fix: `hybrid_nobottleneck`

`fusion_type="hybrid_nobottleneck"` (in `src/models/lstformer_ae.py` /
`tensorflow_version/src/models/lstformer_ae.py`) keeps Track B's (Transformer)
full per-timestep sequence — never pooled to one vector — and injects Track
A's (LSTM) single latent vector as a per-timestep conditioning bias added
to every position, instead of forcing both branches through the same
bottleneck. This is still a genuine hybrid (both branches still contribute,
and the LSTM branch still captures local micro-rhythm information the
Transformer branch might miss), just fused differently.

Result on the same 27-user subset:

| Comparison | Metric | Baseline | hybrid_nobottleneck | p-value | Effect size |
|---|---|---|---|---|---|
| vs. LSTM | EER | 0.490 | **0.372** | **0.0001** (significant) | medium (d=−0.54) |
| vs. LSTM | AUC | 0.466 | **0.644** | **<0.0001** (significant) | medium–large (d=0.70) |
| vs. Transformer | EER | 0.374 | 0.372 | 0.889 (not significant) | negligible |
| vs. Transformer | AUC | 0.655 | 0.644 | 0.471 (not significant) | negligible |

LST-Former now **significantly outperforms the LSTM baseline** and is
**statistically indistinguishable from the Transformer** on both EER and
AUC. This is now the default architecture (both frameworks); the original
`concat` fusion and the other three variants are kept in the codebase
(selectable via `fusion_type`) purely so this ablation study is reproducible.

Full artifacts: `outputs/lstformer_ablation.json` / `.md` (fusion-strategy
sweep), plus the full-scale validation results below.

### 7.4 Full-scale validation results (with the fix applied)

Full re-run with `hybrid_nobottleneck` as the default LST-Former:
- **PyTorch, GPU, all 3 models re-validated at the full dataset size**
  (`python main.py --model all --max-users 100 --device cuda`) — the
  dataset directory holds **100** user files, but the loader
  (`load_all_users`) only produces at least one sliding window for **96**
  of them; 4 files (`130667`, `241253`, `362964`, `379149`) have too few
  keystrokes to form even a single 50-keystroke window and are silently
  dropped before `--max-users` selection ever runs. Of those 96 loadable
  users, **88** have enough windows to reach `BATCH_SIZE` and are actually
  trained/evaluated. So requesting `--max-users 100` reproduces the same
  96-loadable/88-valid set as the earlier `--max-users 96` run — 96 (not
  100) is the true ceiling for this dataset, confirmed by explicitly
  requesting more users than exist in usable form. **The numbers below are
  the final, seeded (`RANDOM_SEED=42`) run** — see Section 7.6(b) for why a
  fixed seed was added and why it matters (results were not reproducible
  run-to-run before this).
- **TensorFlow, CPU, first 20 users** (`tensorflow_version`, `python main.py --model lstformer --max-users 20`) — 18 valid. This TF cross-check
  **predates the `RANDOM_SEED` fix** (seeding was added afterward and only
  re-validated on the PyTorch side, for time reasons); treat it as
  directionally consistent corroboration rather than an exact-reproducibility
  match to the PyTorch numbers.

| Model | PyTorch mean EER (n=88) | PyTorch mean AUC | TF mean EER (n=18) | TF mean AUC |
|---|---|---|---|---|
| LSTM (baseline) | 0.4607 | 0.5025 | 0.4925 | 0.4623 |
| Transformer (challenger) | 0.3524 | 0.6730 | 0.3547 | 0.6688 |
| **LST-Former (proposed, fixed)** | **0.3257** | **0.7025** | **0.3194** | **0.7217** |

**Statistical significance (paired t-test):**

| Comparison | Metric | PyTorch (n=88) p-value | TF (n=18) p-value |
|---|---|---|---|
| LST-Former vs. LSTM | EER | **0.0000** (d=−0.63, medium) | **0.0003** (d=−0.80, medium) |
| LST-Former vs. LSTM | AUC | **0.0000** (d=0.84, large) | **0.0000** (d=1.08, large) |
| LST-Former vs. Transformer | EER | **0.0458** (sig., d=−0.14, negligible) | 0.1211 (not sig., d=−0.19, negligible) |
| LST-Former vs. Transformer | AUC | **0.0458** (sig., d=0.14, negligible) | 0.0907 (not sig., d=0.26, small) |

**Bottom line for your thesis:** LST-Former **significantly outperforms
the LSTM baseline** on both EER and AUC, in both frameworks (p<0.001,
medium-to-large effect sizes). Against the Transformer challenger, the
seeded PyTorch run (n=88) now shows LST-Former **numerically ahead on every
metric, with the gap just crossing statistical significance (p=0.046 on
both EER and AUC)** — but be careful how you word this: the effect size is
**negligible** (d≈0.14), and earlier (unseeded) runs of the identical
comparison landed anywhere from p=0.20 to p=0.86 depending on the specific
random draw (Section 7.6(b)). Report this as "a small, reproducible-given-
this-seed edge for LST-Former that reaches significance in the largest
sample tested, but with a negligible effect size" — not as "LST-Former
decisively beats Transformer." The honest, defensible summary across every
run attempted (seeded and unseeded, PyTorch and TensorFlow) is: **LST-Former
is consistently competitive with the strongest baseline (Transformer) and
consistently, robustly better than the simpler LSTM baseline** — arrived at
through a documented, evidence-based design fix rather than post-hoc tuning
(Section 7.1–7.3).

Resource-scaling (Section 3.9.2) with the fixed architecture still confirms
the theoretical asymptotic complexity: LST-Former remains **O(N²)** overall
(dominated by its Transformer branch), with R² > 0.99 for the latency fit in
both frameworks — the fix changed *how* the branches are fused, not their
individual computational cost.

Full artifacts from all runs are in `outputs/` and
`tensorflow_version/outputs/` (`results.json`, `statistical_analysis_eer.json`,
`statistical_analysis_auc.json`, `resource_scaling.json`, plots,
`asymptotic_report.md`, `lstformer_ablation.json`/`.md`) — safe to inspect,
cite, or overwrite by re-running.

### 7.5 Three-way cross-check: PyTorch GPU vs. TensorFlow CPU vs. TensorFlow GPU

With WSL2 set up (Section 2.7), the same 20-user cross-check was run a third
way — TensorFlow, but with genuine CUDA acceleration this time
(`tensorflow_version/outputs/results_gpu.json`, kept separate from the CPU
run's `results.json` via `--output`) — giving three independent
hardware/framework combinations to check the finding against:

| Model | PyTorch, GPU (n=88) | TensorFlow, CPU (n=18) | TensorFlow, GPU (n=18) |
|---|---|---|---|
| LSTM EER / AUC | 0.462 / 0.501 | 0.493 / 0.462 | 0.495 / 0.464 |
| Transformer EER / AUC | 0.351 / 0.675 | 0.355 / 0.669 | 0.350 / 0.675 |
| **LST-Former EER / AUC** | **0.326 / 0.701** | **0.319 / 0.722** | **0.315 / 0.738** |

**Statistical significance, TensorFlow GPU run (n=18):**
- LST-Former vs. LSTM: EER p=0.0002 (significant, large effect d=−0.82); AUC p<0.0001 (significant, large effect d=1.09)
- LST-Former vs. Transformer: EER p=0.376 (not significant); AUC p=0.111 (not significant) — numerically ahead on both, same pattern as the other two runs, just with a smaller sample so less power to reach significance.

LST-Former beats Transformer's mean EER and AUC in **all three**
independent runs (different hardware, different frameworks, overlapping but
not identical user subsets), and significantly beats LSTM in all three. This
is about as strong a robustness check as is practical to obtain here — the
fix generalizes, it's not a fluke of one framework/hardware combination.

*(Note: this three-way cross-check predates the `RANDOM_SEED` fix in
Section 7.6(b) — none of these three runs were seeded. The "PyTorch, GPU"
column's Transformer number here (0.351/0.675) is noticeably closer to the
final seeded Section 7.4 number (0.3524/0.6730) than to the very first
unseeded run (0.3316/0.6977) quoted earlier in Section 7.1 — one more data
point suggesting ~0.33 was an unusually low/lucky draw rather than
Transformer's "true" typical EER on this data. Take the exact numbers here
as directionally corroborating rather than bit-for-bit reproducible.)*

Resource-scaling was also re-run on WSL2 GPU
(`tensorflow_version/outputs/resource_scaling.json`, N up to 200): the same
O(N)/O(N²)/O(N²) complexity ordering holds, though the R² for the
Transformer's latency fit is noticeably noisier on GPU (0.82 vs. >0.98 on
CPU) — at these tiny model sizes, GPU kernel-launch/dispatch overhead
dominates over actual compute, so the quadratic *attention* term is a
smaller fraction of total latency than on CPU. This is expected and doesn't
change the asymptotic classification (still confirmed by the parameter-count
and memory fits, both R²>0.99).

### 7.6 Chasing a clear win over Transformer: two more angles tried

`hybrid_nobottleneck` made LST-Former statistically *tied* with Transformer,
not clearly ahead of it. Two follow-up ideas were tested to see if the tie
could be broken:

**(a) Genuinely local fusion (`fusion_type="local_injection"`).** Hypothesis:
`hybrid_nobottleneck` still only injects Track A (LSTM) as *one* global
summary vector broadcast identically to every timestep — not really "local"
information anymore, and plausibly redundant with what self-attention
already sees. `local_injection` instead uses the LSTM's per-timestep hidden
state (not just its final one) and injects each position's local state into
the *matching* position of Track B's sequence. Tested on the same 27-user
subset (`python ablate_lstformer.py --max-users 30 --device cuda
--fusion-types local_injection`):

| Variant | Mean EER | Mean AUC | vs. Transformer (EER p) |
|---|---|---|---|
| hybrid_nobottleneck | 0.372 | 0.644 | 0.889 |
| local_injection | 0.383 | 0.624 | 0.912 |

No improvement — if anything marginally worse on this run, still tied with
Transformer either way. This is useful negative evidence: the remaining gap
isn't about *how* the LSTM branch's signal reaches the Transformer branch
(global broadcast vs. per-position). It points at a more fundamental
explanation instead — the Transformer branch alone may already be
extracting most of what's learnable from this HT/FT feature representation,
leaving little unique signal for a second branch to add regardless of fusion
style. `local_injection` is kept in `src/models/lstformer_ae.py`
(`fusion_type="local_injection"`) for reproducibility but is **not** the
default — `hybrid_nobottleneck` remains config's `LSTFORMER_FUSION_TYPE`.

**(b) Architecture-specific training tuning (Section 3.5.2 revision) — tested,
found inconclusive under closer scrutiny, reverted, and used to surface a
real reproducibility gap.** The thesis's original Section 3.5.2 applies one
flat learning rate (`0.001`) and plain Adam to all three architectures
equally. The concern raised: that's not necessarily a fair "equal playing
field" — LSTMs and Transformers have very different loss landscapes, and
attention-based models are well known to sometimes need an LR warm-up phase
(Vaswani et al., 2017), while plain LSTMs are more prone to occasional
gradient-norm spikes that clipping guards against. Both ideas were
implemented (`config.USE_LR_WARMUP`, `config.LSTM_GRAD_CLIP_NORM`,
`src/train.py` / `tensorflow_version/src/train.py`) and tested on the full
88-user run — first without a warmup, then with one added:

| Run | LSTM (+ clip) EER/AUC | Transformer (+warmup) EER/AUC | LST-Former (+warmup) EER/AUC |
|---|---|---|---|
| Before (flat LR, no seed) | 0.4602 / 0.5031 | 0.3316 / 0.6977 | 0.3233 / 0.7007 |
| After adding warmup (no seed) | 0.4582 / 0.5052 | **0.3482 / 0.6815** | 0.3313 / 0.6965 |

Transformer looked clearly worse with warmup added, so the initial
conclusion was "warmup hurts attention models here, revert it." But neither
run used a fixed random seed — re-running identical LSTM config (grad-clip
only, unchanged across runs) already drifted by ~0.004 EER purely from
random init/shuffling, so a same-code re-run was needed to check whether
the Transformer "regression" was a genuine warmup effect or just landing on
a noisy draw. **That re-run (warmup reverted, still no seed) came back at
EER 0.3487 — essentially unchanged from the with-warmup run**, not back
down near the original 0.3316. That's the tell: Transformer's EER varies by
as much as 0.33-0.35+ from run to run on pure random-seed noise alone,
regardless of warmup. The original "warmup hurts" conclusion was likely
confounded by comparing one lucky low-noise baseline run against ordinary
higher-variance runs, not a clean causal effect.

This surfaced a more important, previously-hidden problem than the warmup
question itself: **the training pipeline had no fixed random seed**, so
"the" EER/AUC for a given architecture wasn't actually a single
well-defined number — it was one draw from a distribution with real spread.
That's a reproducibility gap worth fixing regardless of the warmup
question. `main.py` / `ablate_lstformer.py` / `tensorflow_version/main.py`
now seed Python/NumPy/PyTorch (or TensorFlow) and enable deterministic
cuDNN via `config.RANDOM_SEED` (default `42`) at the start of every
architecture's run, so re-running with identical config now reproduces
identical results.

**Final decision: `USE_LR_WARMUP = False` by default.** Not because warmup
was conclusively shown to hurt (the seeded evidence doesn't cleanly support
that causal claim — see above), but because it showed no reproducible
benefit either, adds complexity, and the original mechanistic concern (a
fixed warmup length can eat a large share of a short, early-stopped
per-user training run) remains a plausible risk that a full multi-seed
study would be needed to rule out, which was out of scope here. The option
and this investigation are kept in the code (`config.USE_LR_WARMUP`) rather
than deleted, precisely so this reasoning is reproducible. **Gradient
clipping for the LSTM baseline is kept unconditionally**
(`LSTM_GRAD_CLIP_NORM`, default `1.0`) since it showed no downside in any
run. Suggested wording for the thesis's Section 3.5.2:

> *While the Adam optimizer and initial learning rate of 0.001 are held
> constant across all three architectures, the LSTM baseline additionally
> uses gradient-norm clipping (max norm 1.0) to guard against the occasional
> gradient spikes vanilla LSTMs are prone to. A learning-rate warm-up phase
> for the attention-based models — standard practice for self-attention
> architectures, motivated by the concern that a flat learning rate may not
> constitute an equally fair starting point across architectures with very
> different loss landscapes — was also implemented and evaluated. Under
> controlled, seeded re-runs, its effect could not be reliably distinguished
> from ordinary run-to-run training variance, and it was not adopted for the
> reported results. All reported experiments use a fixed random seed (42)
> across the Python, NumPy, and framework (PyTorch/TensorFlow) random number
> generators to ensure that results are exactly reproducible given identical
> configuration.*

All results in Sections 7.4/7.5 below and in `outputs/results.json` /
`statistical_analysis_*.json` are from the final, seeded (`RANDOM_SEED=42`),
grad-clip-only-for-LSTM, no-warmup configuration — this is the one and only
version of the numbers you should cite. The intermediate no-seed runs
(with and without warmup) were superseded and are only summarized in the
table above for transparency about the investigation; their raw files were
not kept.

### 7.7 Centerpiece: architecture × window-length stability matrix

**Canonical population for all primary thesis tables: n=81**
(`outputs/common_users_81.txt` — users who succeed for every model at
N=25, 50, and 100). See `outputs/CANONICAL_PROTOCOL.md` for the full
lock-in (why not n=88 / n=27, what is primary vs exploratory).

This is the experiment your thesis already planned (“how does input-sequence
length affect reconstruction error and classification stability?”), expanded
into the Results centerpiece. Run:

```powershell
python window_stability.py --max-users 96 --lengths 25 50 100 --device cuda
python analyze_window_stability.py   # paired trajectories + merged efficiency
python investigate_delta_eer.py      # per-user Delta EER heterogeneity
```

Artifacts: `outputs/window_stability.json`, `window_stability.md`,
`window_stability_analysis.json`, `delta_eer_analysis.md`. Framing notes:
`outputs/thesis_framing_notes.md`.

**Common-user EER trajectory (n=81 — PRIMARY):**

| Window N | LSTM | Transformer | LST-Former | vs Transformer | vs LSTM |
| ---: | ---: | ---: | ---: | --- | --- |
| 25 | 0.440 | 0.370 | **0.327** | p=0.003, d=−0.34 (small) | p≈0, d=−0.79 (large) |
| 50 | 0.451 | 0.345 | **0.311** | p=0.014, d=−0.28 (small) | p≈0, d=−0.91 (large) |
| 100 | 0.455 | 0.332 | **0.318** | p=0.302, d=−0.12 (n.s.) | p≈0, d=−0.93 (large) |

**Locked interpretation:** measurable advantage over LSTM (large effect) at
all N; smaller, sequence-length-dependent edge over Transformer at short/
medium N that disappears at N=100; benefit depends on fusion design
(`hybrid_nobottleneck`), not “any LSTM+Transformer combo”; heterogeneous
per-user gains; higher compute cost → security–efficiency trade-off — **not**
universal superiority.

---

## 7.8 Publishing this repo to GitHub

```powershell
cd <this-repo>
git init
git add .
git status   # confirm dataset / .venv / *.pt are NOT staged
git commit -m "Initial commit: keystroke LST-Former thesis code and results"
git branch -M main
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main
```

`.gitignore` excludes: virtualenvs, Clarkson/Buffalo data, `*.pt` checkpoints,
thesis PDF drafts, and zip archives. **Do** commit `outputs/*.json`,
`outputs/*.md`, and `outputs/plots/` so readers can inspect the locked
n=81 tables without re-training.

---

## 8. Known limitations / honest caveats

- **Dataset not in git**: clones must obtain Clarkson II separately (§2.0).
- **Per-user models on limited data**: some users in the dataset have too
  few keystrokes to reach `BATCH_SIZE` training windows and are
  automatically skipped (logged as `"status": "skipped"` in `results.json`).
- **CPU-only memory proxy (only if no GPU is present)**: `resource_scaling.py`
  and `main.py` use true VRAM (`torch.cuda.max_memory_allocated`) when a CUDA
  GPU is available. If run on a machine without an NVIDIA GPU, memory falls
  back to a noisier host-RAM proxy via `psutil`, which is indicative only,
  not literal VRAM.
- **Framework**: the root implementation uses PyTorch; a full TensorFlow
  2.16.1/Keras port matching Section 3.7.2 now lives in `tensorflow_version/`
  (see Section 2.4-2.6 above). TensorFlow has **no GPU support natively on
  Windows** (a TensorFlow/Windows platform limitation since TF 2.11, not
  something specific to this project — see Section 2.5). This has been
  worked around via WSL2 (Section 2.7), which now gives genuine
  CUDA-accelerated TensorFlow numbers alongside the native-Windows CPU-only
  ones (Section 7.5) — both are available and documented.
- **Hyperparameters not explicitly given in the thesis** (hidden/latent
  dimensions, number of attention heads, Transformer depth) use reasonable
  defaults documented in `config.py` — tune them if your results don't look
  representative.
- **Single-seed results**: `RANDOM_SEED=42` (Section 7.6(b)) makes results
  *reproducible* (re-running gives identical numbers), but reproducible is
  not the same as *robust to the choice of seed*. Section 7.6(b) shows
  Transformer's EER varied from ~0.33 to ~0.35 across different unseeded
  runs, and the LST-Former-vs-Transformer significance test result (p=0.046,
  Section 7.4) is from this one seed only. A proper robustness check would
  re-run the full validation under several different seeds and report the
  spread — not done here for time/compute reasons. Treat "statistically
  significant at p=0.046 with a negligible effect size" as a fragile,
  seed-specific result, not a strong claim.
- **Canonical population**: primary claims use **n=81**
  (`outputs/CANONICAL_PROTOCOL.md`). The n=88 @ N=50 run is supplementary.

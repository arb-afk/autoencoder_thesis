"""
Resource-Scaling Analysis (Section 3.9.2 and 3.10.2).

Trains each architecture briefly at several sequence lengths N (default
{25, 50, 100}) and measures:
  - Trainable parameter count (P)
  - Peak memory during training (VRAM if CUDA is available, else host RAM
    via psutil as the closest available proxy on CPU-only machines)
  - CPU-only inference latency (Section 3.9.1)

Then fits the theoretically-expected regression curve to each metric
(linear for LSTM, quadratic for Transformer/LST-Former) and reports the R^2,
which empirically verifies the asymptotic complexity claims in
src/complexity.py.

Usage:
    python resource_scaling.py
    python resource_scaling.py --user 118630 --lengths 25 50 100 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.complexity import THEORETICAL_COMPLEXITY, best_fit, print_theoretical_summary
from src.evaluate import measure_inference_latency
from src.preprocessing import (
    load_keystroke_file,
    extract_ht_ft,
    sliding_windows,
)
from src.train import build_model

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _pick_reference_user(data_dir: Path, min_keystrokes: int = 2000) -> Path:
    """Pick a user file with enough keystrokes to build windows at every N."""
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        events = load_keystroke_file(path)
        if len(events) >= min_keystrokes:
            return path
    raise RuntimeError("No user file with enough keystrokes found for scaling analysis")


def _windows_for_length(path: Path, seq_len: int) -> np.ndarray:
    """Reference windows for latency/memory timing (not authentication eval).

    Uses train-only z-score fit on the chronological 70% feature prefix for
    consistency with the leak-free protocol; windows are built only on that
    training portion.
    """
    from src.preprocessing import split_feature_sequence, zscore_apply, zscore_fit

    events = load_keystroke_file(path)
    features = extract_ht_ft(events)
    train_f, _, _ = split_feature_sequence(features)
    mean, std = zscore_fit(train_f if len(train_f) else features)
    normalized = zscore_apply(train_f if len(train_f) else features, mean, std)
    stride = max(1, seq_len // 2)
    return sliding_windows(normalized, window_size=seq_len, stride=stride)


def _peak_memory_during_training(model, windows, device, n_steps=5) -> float | None:
    """
    Runs a few forward+backward steps and returns peak memory in MB.
    Uses CUDA peak allocator stats if available, otherwise psutil RSS delta
    as a CPU proxy. Returns None if neither is available.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    batch_size = min(config.BATCH_SIZE, len(windows))
    batch = torch.as_tensor(windows[:batch_size], dtype=torch.float32).to(device)

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        for _ in range(n_steps):
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)

    if _HAS_PSUTIL:
        process = psutil.Process()
        baseline = process.memory_info().rss
        peak = baseline
        for _ in range(n_steps):
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            peak = max(peak, process.memory_info().rss)
        return (peak - baseline) / (1024 * 1024)

    return None


def run_scaling_analysis(
    lengths: list[int],
    user_path: Path,
    device: str = "cpu",
) -> dict:
    models = ["lstm", "transformer", "lstformer"]
    raw_results: dict[str, dict[str, list]] = {
        m: {"N": [], "params": [], "latency_ms": [], "memory_mb": []} for m in models
    }

    for seq_len in lengths:
        print(f"\n--- Sequence length N={seq_len} ---")
        windows = _windows_for_length(user_path, seq_len)
        if len(windows) < 2:
            print(f"  Not enough data to build windows at N={seq_len}, skipping")
            continue

        for model_name in models:
            model = build_model(model_name, seq_len=seq_len)
            num_params = sum(p.numel() for p in model.parameters())

            mem_mb = _peak_memory_during_training(model, windows, device)

            sample = torch.randn(1, seq_len, config.FEATURE_DIM)
            latency_ms = measure_inference_latency(model, sample, device="cpu")

            raw_results[model_name]["N"].append(seq_len)
            raw_results[model_name]["params"].append(num_params)
            raw_results[model_name]["latency_ms"].append(latency_ms)
            raw_results[model_name]["memory_mb"].append(mem_mb)

            mem_str = f"{mem_mb:.2f} MB" if mem_mb is not None else "n/a"
            print(
                f"  {model_name:12s} | params={num_params:>8,} | "
                f"latency={latency_ms:6.3f} ms | peak_mem={mem_str}"
            )

    # Empirical regression fits (Section 3.10.2)
    fits: dict[str, dict] = {}
    for model_name in models:
        data = raw_results[model_name]
        if len(data["N"]) < 2:
            continue
        fits[model_name] = {
            "latency_fit": best_fit(model_name, data["N"], data["latency_ms"]),
            "params_fit": best_fit(model_name, data["N"], data["params"]),
        }
        mem_values = [m for m in data["memory_mb"] if m is not None]
        if len(mem_values) == len(data["N"]) and len(mem_values) >= 2:
            fits[model_name]["memory_fit"] = best_fit(model_name, data["N"], mem_values)

    return {"raw": raw_results, "fits": fits}


def save_plots(results: dict, out_dir: Path) -> list[Path]:
    """
    Render one figure per metric (latency, params, memory) with all three
    models' measured points and fitted curves overlaid, so the asymptotic
    behavior can be visually "recorded" (e.g. pasted into the thesis as a
    figure for Section 3.9.2 / 3.10.2).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    colors = {"lstm": "tab:blue", "transformer": "tab:orange", "lstformer": "tab:green"}

    metrics = [
        ("latency_ms", "latency_fit", "Inference Latency (ms)", "latency"),
        ("params", "params_fit", "Trainable Parameters", "params"),
        ("memory_mb", "memory_fit", "Peak Memory (MB)", "memory"),
    ]

    for raw_key, fit_key, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(7, 5))
        plotted_any = False

        for model_name, data in results["raw"].items():
            xs = np.array(data["N"], dtype=float)
            ys_raw = data[raw_key]
            if any(v is None for v in ys_raw) or len(xs) < 2:
                continue
            ys = np.array(ys_raw, dtype=float)
            plotted_any = True

            ax.scatter(xs, ys, color=colors[model_name], label=f"{model_name} (measured)")

            fit = results["fits"].get(model_name, {}).get(fit_key)
            if fit:
                x_smooth = np.linspace(xs.min(), xs.max(), 100)
                y_smooth = np.polyval(fit["coeffs"], x_smooth)
                ax.plot(
                    x_smooth,
                    y_smooth,
                    color=colors[model_name],
                    linestyle="--",
                    label=f"{model_name} fit ({fit['expected_order']}, R\u00b2={fit['r2']:.3f})",
                )

        if not plotted_any:
            plt.close(fig)
            continue

        ax.set_xlabel("Sequence Length N (keystrokes per window)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs. Sequence Length N")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        path = out_dir / f"scaling_{filename}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    return saved


def save_markdown_report(results: dict, plot_paths: list[Path], out_path: Path) -> None:
    """
    Write a self-contained Markdown report of the asymptotic complexity
    findings (theoretical + empirical), suitable for pasting into the
    thesis Results chapter or an appendix.
    """
    lines = [
        "# Asymptotic Complexity Analysis",
        "",
        "Generated by `resource_scaling.py` (Sections 3.9.2 and 3.10.2).",
        "",
        "## 1. Theoretical Complexity",
        "",
        "| Model | Time | Space | Rationale |",
        "|---|---|---|---|",
    ]
    for name, info in THEORETICAL_COMPLEXITY.items():
        lines.append(f"| {name} | {info['time']} | {info['space']} | {info['explanation']} |")

    lines += [
        "",
        "## 2. Empirical Scaling (consistent with expected trends — not a proof)",
        "",
        "Hardware latency/memory include constant overhead (kernel launch,",
        "memory bandwidth, parallelization). High R² for a linear/quadratic",
        "regression shows measurements are **consistent with** the expected",
        "growth order; it does **not** prove mathematical O(N) / O(N²).",
        "Prefer the log-log empirical exponent p (y ∝ N^p) as the primary",
        "empirical summary.",
        "",
    ]
    for model_name, model_fits in results["fits"].items():
        lines.append(f"### {model_name.upper()}")
        lines.append("")
        lines.append("| Metric | Expected | Fitted equation | R² | log-log p | best candidate |")
        lines.append("|---|---|---|---|---|---|")
        for metric_name, fit in model_fits.items():
            diag = fit.get("scaling_diagnostics", {})
            ll = diag.get("log_log", {})
            p = ll.get("p")
            p_str = f"{p:.3f}" if isinstance(p, float) and p == p else "n/a"
            best = diag.get("best_by_r2", "n/a")
            lines.append(
                f"| {metric_name} | {fit['expected_order']} | `{fit['form']}` | "
                f"{fit['r2']:.4f} | {p_str} | {best} |"
            )
        lines.append("")
        # claim line
        lat = model_fits.get("latency_fit", {})
        if lat.get("claim"):
            lines.append(f"*{lat['claim']}*")
            lines.append("")

    if plot_paths:
        lines += ["## 3. Figures", ""]
        for path in plot_paths:
            lines.append(f"![{path.stem}]({path.name})")
            lines.append("")

    lines += [
        "## 4. Raw Measurements",
        "",
    ]
    for model_name, data in results["raw"].items():
        lines.append(f"### {model_name.upper()}")
        lines.append("")
        lines.append("| N | Params | Latency (ms) | Peak Memory (MB) |")
        lines.append("|---|---|---|---|")
        for i, n in enumerate(data["N"]):
            mem = data["memory_mb"][i]
            mem_str = f"{mem:.2f}" if mem is not None else "n/a"
            lines.append(f"| {n} | {data['params'][i]:,} | {data['latency_ms'][i]:.4f} | {mem_str} |")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Resource-scaling analysis (Section 3.9.2)")
    parser.add_argument("--lengths", type=int, nargs="+", default=config.SEQUENCE_LENGTHS)
    parser.add_argument("--user", type=str, default=None, help="Specific user file to use")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for the training/memory measurement (latency is always CPU-only, "
        "per Section 3.7.1). Defaults to cuda if available, else cpu.",
    )
    args = parser.parse_args()
    train_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training/memory device: {train_device} (inference latency is always measured on CPU)")

    print_theoretical_summary()

    if args.user:
        user_path = config.DATA_DIR / args.user
    else:
        print(f"\nPicking a reference user with enough keystrokes for N up to {max(args.lengths)}...")
        user_path = _pick_reference_user(config.DATA_DIR, min_keystrokes=max(args.lengths) * 4)
    print(f"Reference user file: {user_path.name}")

    results = run_scaling_analysis(args.lengths, user_path, device=train_device)

    print("\n" + "=" * 75)
    print("Empirical fits (consistent with theoretical orders - not a proof)")
    print("=" * 75)
    for model_name, model_fits in results["fits"].items():
        print(f"\n{model_name.upper()} (expected: {model_fits['latency_fit']['expected_order']})")
        for metric_name, fit in model_fits.items():
            ll = fit.get("scaling_diagnostics", {}).get("log_log", {})
            p = ll.get("p")
            p_str = f"  |  log-log p ~ {p:.3f}" if isinstance(p, float) and p == p else ""
            print(f"  {metric_name}: {fit['form']}  |  R^2 = {fit['r2']:.4f}{p_str}")
            claim = fit.get("claim", "").replace("²", "^2").replace("—", "-")
            print(f"    -> {claim}")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.OUTPUT_DIR / "resource_scaling.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    plots_dir = config.OUTPUT_DIR / "plots"
    plot_paths = save_plots(results, plots_dir)
    for p in plot_paths:
        print(f"Plot saved to {p}")

    report_path = config.OUTPUT_DIR / "asymptotic_report.md"
    save_markdown_report(results, plot_paths, report_path)
    print(f"Markdown report saved to {report_path}")
    print("^ this report + the PNGs in outputs/plots/ are what you cite/paste into the thesis.")


if __name__ == "__main__":
    main()

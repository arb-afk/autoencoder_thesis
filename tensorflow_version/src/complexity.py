"""
Asymptotic (Big-O) complexity analysis for the LSTM, Transformer, and
LST-Former autoencoders (Sections 3.9.2 and 3.10.2).

Framework-agnostic -- identical to ../../src/complexity.py.

There are two complementary ways to "find" the asymptotic notation of a model:

1. THEORETICAL (derived from the architecture, no code execution needed):
   Look at what operation dominates each forward pass as a function of the
   window length N, and how its cost grows. See THEORETICAL_COMPLEXITY below.

2. EMPIRICAL (regression on measured data, what the thesis actually asks for
   in Section 3.10.2): run each model at several sequence lengths N, measure
   a resource (inference latency, memory, etc.), and fit a curve to it.
   - LSTM   -> fit a straight line   y = a*N + b       (expect O(N))
   - Trans. -> fit a parabola        y = a*N^2 + b*N + c (expect O(N^2))
   The R^2 (coefficient of determination) of each fit tells you how well
   that order of growth explains the measured data. This is exactly what
   `resource_scaling.py` does using `best_fit()` below.
"""

from __future__ import annotations

import numpy as np

THEORETICAL_COMPLEXITY = {
    "lstm": {
        "time": "O(N)",
        "space": "O(N)",
        "explanation": (
            "The LSTM encoder processes the window one keystroke at a time "
            "through recurrent memory cells. Each of the N steps does a "
            "constant amount of work (gate computations sized by hidden_dim), "
            "so total time and activation memory scale linearly with the "
            "window length N."
        ),
    },
    "transformer": {
        "time": "O(N^2)",
        "space": "O(N^2)",
        "explanation": (
            "Multi-Head Self-Attention computes an N x N attention score "
            "matrix -- every keystroke attends to every other keystroke in "
            "the window -- so both time and memory scale quadratically with "
            "N. The feed-forward sublayers alone would be O(N), but "
            "attention dominates for the window sizes used here."
        ),
    },
    "lstformer": {
        "time": "O(N) + O(N^2) = O(N^2)",
        "space": "O(N) + O(N^2) = O(N^2)",
        "explanation": (
            "Track A (LSTM branch) contributes O(N); Track B (Transformer "
            "branch) contributes O(N^2). The two branches run in parallel "
            "and their costs add, so the quadratic term from the attention "
            "branch dominates asymptotically. In practice the constant "
            "factor is smaller than a deep stand-alone Transformer because "
            "the LST-Former's Transformer branch offloads local rhythm "
            "modeling to the cheaper LSTM branch."
        ),
    },
}


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit y = a*x + b (verifies O(N) growth). Returns coefficients and R^2."""
    coeffs = np.polyfit(x, y, deg=1)
    pred = np.polyval(coeffs, x)
    return {
        "degree": 1,
        "coeffs": coeffs.tolist(),
        "r2": _r_squared(y, pred),
        "form": f"y = {coeffs[0]:.6g}*N + {coeffs[1]:.6g}",
    }


def quadratic_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit y = a*x^2 + b*x + c (verifies O(N^2) growth). Returns coefficients and R^2."""
    coeffs = np.polyfit(x, y, deg=2)
    pred = np.polyval(coeffs, x)
    return {
        "degree": 2,
        "coeffs": coeffs.tolist(),
        "r2": _r_squared(y, pred),
        "form": f"y = {coeffs[0]:.6g}*N^2 + {coeffs[1]:.6g}*N + {coeffs[2]:.6g}",
    }


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1 - ss_res / ss_tot)


def best_fit(model_name: str, x, y) -> dict:
    """
    Fit the theoretically-expected regression for a given model
    (linear for LSTM, quadratic for Transformer/LST-Former), per Section 3.10.2.
    """
    fit_fn = linear_fit if model_name == "lstm" else quadratic_fit
    result = fit_fn(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    result["model"] = model_name
    result["expected_order"] = THEORETICAL_COMPLEXITY[model_name]["time"]
    return result


def print_theoretical_summary() -> None:
    print("\nTheoretical Asymptotic Complexity (per forward pass, window length N)")
    print("=" * 75)
    for name, info in THEORETICAL_COMPLEXITY.items():
        print(f"\n{name.upper()}")
        print(f"  Time:  {info['time']}")
        print(f"  Space: {info['space']}")
        print(f"  Why:   {info['explanation']}")

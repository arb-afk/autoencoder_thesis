"""Poll window_stability.json until the 9-cell matrix is complete, then analyze + ablate."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
WS = OUT / "window_stability.json"
MODELS = ("lstm", "transformer", "lstformer")
LENGTHS = (25, 50, 100)
EXPECTED = [f"{m}_N{n}" for n in LENGTHS for m in MODELS]


def ws_process_running() -> bool:
    """True if window_stability.py appears in the process list (Windows)."""
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
             "Select-Object -ExpandProperty CommandLine"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return "window_stability.py" in out


def cell_done(cell: dict, min_users: int = 80) -> bool:
    per = cell.get("per_user", [])
    return len(per) >= min_users


def matrix_complete() -> bool:
    if not WS.exists():
        return False
    d = json.loads(WS.read_text(encoding="utf-8"))
    if not all(k in d for k in EXPECTED):
        return False
    if not all(cell_done(d[k]) for k in EXPECTED):
        return False
    # Prefer waiting until the trainer has exited so the final flush is done.
    return not ws_process_running()


def status_line() -> str:
    if not WS.exists():
        return "no json yet"
    d = json.loads(WS.read_text(encoding="utf-8"))
    parts = []
    for k in EXPECTED:
        n = len(d.get(k, {}).get("per_user", []))
        parts.append(f"{k}={n}")
    return " | ".join(parts)


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    py = sys.executable
    print("Watching for complete leak-free window_stability matrix...", flush=True)
    while not matrix_complete():
        print(time.strftime("%H:%M:%S"), status_line(), flush=True)
        time.sleep(120)

    print("Matrix complete. Post-processing...", flush=True)
    run([py, str(ROOT / "analyze_window_stability.py")])
    run([py, str(ROOT / "investigate_delta_eer.py")])

    users_file = OUT / "common_users_81.txt"
    n = len([ln for ln in users_file.read_text(encoding="utf-8").splitlines() if ln.strip()])
    print(f"Common users: {n} in {users_file}", flush=True)

    ablate_out = OUT / "lstformer_ablation_aligned_N50.json"
    run(
        [
            py,
            str(ROOT / "ablate_lstformer.py"),
            "--users-file",
            str(users_file),
            "--window-size",
            "50",
            "--fusion-types",
            "concat",
            "local_injection",
            "hybrid_nobottleneck",
            "--baseline-from-window-stability",
            "--output",
            str(ablate_out),
            "--device",
            "cuda",
        ]
    )
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

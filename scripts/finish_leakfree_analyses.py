"""Run after window_stability.py finishes under the leak-free protocol.

Regenerates paired analyses, ΔEER report, and prints the command for the
aligned fusion ablation on the recomputed common-user set.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    py = sys.executable
    steps = [
        [py, str(ROOT / "analyze_window_stability.py")],
        [py, str(ROOT / "investigate_delta_eer.py")],
    ]
    for cmd in steps:
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)

    users = ROOT / "outputs" / "common_users_81.txt"
    n = len([ln for ln in users.read_text(encoding="utf-8").splitlines() if ln.strip()]) if users.exists() else 0
    print(f"\nCommon-user list: {users} ({n} users)")
    print(
        "Next (aligned fusion ablation):\n"
        f"  {py} ablate_lstformer.py "
        f"--users-file outputs/common_users_81.txt "
        "--window-size 50 --fusion-types concat local_injection hybrid_nobottleneck "
        "--baseline-from-window-stability "
        "--output outputs/lstformer_ablation_aligned_N50.json --device cuda"
    )


if __name__ == "__main__":
    main()

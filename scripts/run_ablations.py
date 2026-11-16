"""Sprint 4 ablations: run one change at a time and log each with a variant tag.

    python scripts/run_ablations.py bands      # frequency-band sweep (where does the information sit?)
    python scripts/run_ablations.py ea         # Euclidean Alignment (helps new people?)
    python scripts/run_ablations.py crops      # cropped training (more examples per trial)
    python scripts/run_ablations.py all

Each run goes through scripts/run_experiment.py, so every result is logged and the usual safety
locks apply. Protocols: Track 3 uses the official validation split (never the test split here)
plus LOSO; Nieto uses within-subject, cross-session and LOSO.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_experiment.py"

BANDS = [(1.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 40.0)]
TRACK3_PROTOCOLS = ["predefined_val", "loso"]
NIETO_PROTOCOLS = ["within_subject", "cross_session", "loso"]


def run(*args: str) -> int:
    cmd = [sys.executable, "-W", "ignore", "-u", str(RUNNER), *args]
    print("+", " ".join(args), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def bands() -> None:
    for lo, hi in BANDS:
        tag = f"band{lo:g}-{hi:g}"
        run("--dataset", "track3", "--model", "riemann", "--protocol", *TRACK3_PROTOCOLS,
            "--l-freq", str(lo), "--h-freq", str(hi), "--variant", tag)
        run("--dataset", "nieto", "--model", "riemann", "--protocol", *NIETO_PROTOCOLS,
            "--l-freq", str(lo), "--h-freq", str(hi), "--variant", tag)


def ea() -> None:
    run("--dataset", "track3", "--model", "riemann_ea", "--protocol", *TRACK3_PROTOCOLS, "--h-freq", "30", "--variant", "ea")
    run("--dataset", "nieto", "--model", "riemann_ea", "--protocol", *NIETO_PROTOCOLS, "--variant", "ea")


def crops() -> None:
    # 1 s windows every 0.5 s: 5 crops per 3.1 s Track 3 trial, 4 per 2.5 s Nieto trial
    run("--dataset", "track3", "--model", "riemann", "--protocol", *TRACK3_PROTOCOLS, "--h-freq", "30",
        "--crop", "1.0", "0.5", "--variant", "crops")
    run("--dataset", "nieto", "--model", "riemann", "--protocol", *NIETO_PROTOCOLS, "--crop", "1.0", "0.5", "--variant", "crops")


def main(argv: list[str]) -> int:
    steps = {"bands": bands, "ea": ea, "crops": crops}
    which = argv or ["all"]
    if which == ["all"]:
        which = list(steps)
    for name in which:
        if name not in steps:
            print(f"unknown ablation {name!r}; choose from {list(steps)} or all", file=sys.stderr)
            return 2
        steps[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

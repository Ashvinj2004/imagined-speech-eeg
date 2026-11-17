"""Reproduce every headline number from a fresh clone, in order.

    python scripts/reproduce.py --list          # show the steps
    python scripts/reproduce.py                 # run them all (many hours on CPU)
    python scripts/reproduce.py main            # only the main table (no neural networks, no ablations)

Prerequisites: the README setup and `python scripts/download_data.py`. Each step calls
scripts/run_experiment.py, so every number is logged to results/LOG.md; afterwards
`python scripts/build_tables.py` rebuilds results/tables/*.md from the log.

Steps that use the official Track 3 test split run with --final: they are the one allowed
touch of that split per configuration, with settings fixed in advance by the validation runs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

MAIN = [
    # Track 3: settings chosen on validation, then one test run per model
    ["--dataset", "track3", "--model", "majority", "--protocol", "predefined_val"],
    ["--dataset", "track3", "--model", "riemann", "--protocol", "predefined_val"],
    ["--dataset", "track3", "--model", "riemann", "--h-freq", "30", "--protocol", "predefined_val"],
    ["--dataset", "track3", "--model", "majority", "--protocol", "predefined", "--final"],
    ["--dataset", "track3", "--model", "riemann", "--h-freq", "30", "--protocol", "predefined", "--final", "--n-perm", "100"],
    ["--dataset", "track3", "--model", "riemann", "--h-freq", "30", "--protocol", "loso", "--n-perm", "20"],
    # Nieto: all three protocols
    ["--dataset", "nieto", "--model", "majority", "--protocol", "within_subject", "cross_session", "loso"],
    ["--dataset", "nieto", "--model", "riemann", "--protocol", "within_subject", "cross_session", "loso"],
]
DEEP = [
    ["--dataset", "track3", "--model", "eegnet", "--protocol", "predefined_val"],
    ["--dataset", "track3", "--model", "shallow", "--protocol", "predefined_val"],
    ["--dataset", "track3", "--model", "eegnet", "--protocol", "predefined", "--final"],
    ["--dataset", "track3", "--model", "shallow", "--protocol", "predefined", "--final"],
    ["--dataset", "nieto", "--model", "shallow", "--protocol", "within_subject"],
]
GROUPS = {"main": MAIN, "deep": DEEP}


def main(argv: list[str]) -> int:
    if argv == ["--list"]:
        for name, steps in GROUPS.items():
            print(f"[{name}]")
            for s in steps:
                print("  run_experiment.py", " ".join(s))
        print("[ablations]\n  run_ablations.py all")
        return 0
    which = argv or ["main", "deep", "ablations"]
    for name in which:
        if name == "ablations":
            rc = subprocess.call([PY, str(ROOT / "scripts" / "run_ablations.py"), "all"], cwd=ROOT)
        elif name in GROUPS:
            rc = 0
            for step in GROUPS[name]:
                print("+ run_experiment.py", " ".join(step), flush=True)
                rc |= subprocess.call([PY, "-W", "ignore", "-u", str(ROOT / "scripts" / "run_experiment.py"), *step], cwd=ROOT)
        else:
            print(f"unknown group {name!r}; choose from main, deep, ablations", file=sys.stderr)
            return 2
        if rc:
            print(f"group {name} had failures (exit {rc})", file=sys.stderr)
            return rc
    subprocess.call([PY, str(ROOT / "scripts" / "build_tables.py")], cwd=ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

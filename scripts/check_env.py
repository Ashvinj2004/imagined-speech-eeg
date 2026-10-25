"""Sprint 1 gate check: every teammate runs this and compares the output.

    python scripts/check_env.py

All four laptops should print the same Python minor version, the same package versions
(from requirements.lock) and the same dataset shape/summary. Exits non-zero if a core
import fails.
"""

from __future__ import annotations

import importlib
import platform
import sys

PACKAGES = ["numpy", "scipy", "pandas", "sklearn", "mne", "pyriemann", "torch", "braindecode", "moabb", "bci"]


def main() -> int:
    print(f"python   {platform.python_version()} ({platform.system()} {platform.machine()})")
    failed = []
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"{name:<12} {getattr(mod, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001 - report every failure, e.g. DLLs blocked by Windows Smart App Control
            print(f"{name:<12} FAILED: {type(e).__name__}: {e}")
            failed.append(name)

    if "bci" not in failed:
        from bci.data import load

        ds = load("synthetic", seed=0)
        print(f"\nX shape  {ds.X.shape}  dtype {ds.X.dtype}  first value {ds.X[0, 0, 0]:.6f}")
        print(ds.summary())

    if failed:
        print(f"\nFAILED imports: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\nOK: compare this output with your teammates'")
    return 0


if __name__ == "__main__":
    sys.exit(main())

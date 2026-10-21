"""End-to-end plumbing check on SYNTHETIC data. These numbers are never reported or logged.

Exits non-zero unless:
  1. majority baseline stays at or below the chance threshold under every protocol,
  2. the Riemannian baseline beats the chance threshold within-subject,
  3. Riemannian LOSO accuracy < Riemannian within-subject accuracy.
"""

from __future__ import annotations

import sys

import pandas as pd

from bci.data import load
from bci.eval.harness import evaluate, summarize
from bci.models import majority, riemann_ts_lr

SEED = 0


def main() -> int:
    ds = load("synthetic", seed=SEED)
    print(ds.summary(), "\n")

    folds = []
    for name, factory in [("majority", majority), ("riemann_ts_lr", lambda: riemann_ts_lr(seed=SEED))]:
        df = evaluate(factory, ds, seed=SEED)
        df.insert(0, "model", name)
        folds.append(df)
    folds = pd.concat(folds, ignore_index=True)
    summary = summarize(folds, n_classes=ds.n_classes)

    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False), "\n")

    acc = summary.set_index(["model", "protocol"])
    failures = []
    maj = acc.loc["majority"]
    if (maj["acc"] > maj["chance_threshold"]).any():
        failures.append("majority baseline exceeded the chance threshold")
    rie = acc.loc["riemann_ts_lr"]
    if not rie.loc["within_subject", "acc"] > rie.loc["within_subject", "chance_threshold"]:
        failures.append("Riemannian model did not beat chance within-subject")
    if not rie.loc["loso", "acc"] < rie.loc["within_subject", "acc"]:
        failures.append("Riemannian LOSO accuracy is not below within-subject accuracy")

    if failures:
        for f in failures:
            print(f"SMOKE TEST FAILED: {f}", file=sys.stderr)
        return 1
    print("SMOKE TEST PASSED (synthetic data: plumbing check only, do not report)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

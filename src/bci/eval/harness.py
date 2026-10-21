"""The only sanctioned way to score a model.

`evaluate` takes a zero-arg model FACTORY, not a model: a fresh, unfitted estimator is built
for every fold, so nothing learned on one fold (scalers, covariance means, early-stopping
state) can leak into another.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from bci.data.base import EpochsDataset
from bci.eval.chance_level import binomial_pvalue, chance_threshold
from bci.eval.splits import get_folds

DEFAULT_PROTOCOLS = ("within_subject", "cross_session", "loso")
LEAKAGE_SUSPICION_ACC = 0.9


def evaluate(
    make_model: Callable[[], object],
    ds: EpochsDataset,
    protocols: Iterable[str] = DEFAULT_PROTOCOLS,
    seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """One row per fold: protocol, held_out, n_train, n_test, acc, bal_acc, 1/k, chance threshold, binomial p."""
    k = ds.n_classes
    rows = []
    for protocol in protocols:
        for fold in get_folds(ds, protocol, seed=seed):
            model = make_model()
            model.fit(ds.X[fold.train_idx], ds.y[fold.train_idx])
            y_true = ds.y[fold.test_idx]
            y_pred = np.asarray(model.predict(ds.X[fold.test_idx]))
            n_test = len(y_true)
            n_correct = int((y_pred == y_true).sum())
            rows.append(
                {
                    "protocol": protocol,
                    "held_out": fold.held_out,
                    "subject": fold.subject,
                    "n_train": len(fold.train_idx),
                    "n_test": n_test,
                    "n_correct": n_correct,
                    "acc": n_correct / n_test,
                    "bal_acc": balanced_accuracy_score(y_true, y_pred),
                    "one_over_k": 1.0 / k,
                    "chance_threshold": chance_threshold(n_test, k, alpha),
                    "binom_p": binomial_pvalue(n_correct, n_test, k),
                }
            )
    df = pd.DataFrame(rows)
    df.attrs["n_classes"] = k
    df.attrs["seed"] = seed
    df.attrs["alpha"] = alpha
    return df


def summarize(df: pd.DataFrame, n_classes: int | None = None, alpha: float | None = None) -> pd.DataFrame:
    """Pool folds per protocol (and per model, if a `model` column exists).

    Each trial is tested exactly once per protocol, so pooling correct counts over folds gives
    one accuracy over all trials, judged against the chance threshold for that pooled n_test.
    """
    k = n_classes or df.attrs.get("n_classes") or int(round(1 / df["one_over_k"].iloc[0]))
    alpha = alpha or df.attrs.get("alpha", 0.05)
    keys = ["model", "protocol"] if "model" in df.columns else ["protocol"]
    out = []
    for key, g in df.groupby(keys, sort=False):
        n_test, n_correct = int(g["n_test"].sum()), int(g["n_correct"].sum())
        per_subj = g.groupby("subject")[["n_correct", "n_test"]].sum()
        per_subj_acc = per_subj["n_correct"] / per_subj["n_test"]
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row.update(
            n_folds=len(g),
            n_test=n_test,
            acc=n_correct / n_test,
            bal_acc_mean=g["bal_acc"].mean(),
            subj_acc_min=per_subj_acc.min(),
            subj_acc_max=per_subj_acc.max(),
            one_over_k=1.0 / k,
            chance_threshold=chance_threshold(n_test, k, alpha),
            binom_p=binomial_pvalue(n_correct, n_test, k),
        )
        row["above_chance"] = row["acc"] > row["chance_threshold"]
        out.append(row)
        if k >= 3 and row["acc"] > LEAKAGE_SUSPICION_ACC:
            msg = (
                f"PROBABLE DATA LEAKAGE: pooled accuracy {row['acc']:.3f} > {LEAKAGE_SUSPICION_ACC} "
                f"on {k} classes for {row}. Stop and investigate before using this result (CLAUDE.md rule 4)."
            )
            print("\n" + "!" * 80 + f"\n{msg}\n" + "!" * 80 + "\n", file=sys.stderr)
            warnings.warn(msg, stacklevel=2)
    return pd.DataFrame(out)


def permutation_null(
    make_model: Callable[[], object],
    ds: EpochsDataset,
    protocol: str,
    n_perm: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Pooled accuracy under `n_perm` label shuffles (within subject), rerunning the whole evaluation each time.

    Compare the real pooled accuracy against this with `bci.eval.chance_level.permutation_pvalue`.
    """
    rng = np.random.default_rng(seed)
    scores = np.empty(n_perm)
    for p in range(n_perm):
        y_perm = ds.y.copy()
        for s in np.unique(ds.subject):
            idx = np.flatnonzero(ds.subject == s)
            y_perm[idx] = rng.permutation(ds.y[idx])
        df = evaluate(make_model, replace(ds, y=y_perm), [protocol], seed=seed)
        scores[p] = df["n_correct"].sum() / df["n_test"].sum()
    return scores

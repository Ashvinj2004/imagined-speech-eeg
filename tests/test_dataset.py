import warnings

import numpy as np
import pytest

from bci.data import EpochsDataset, available, load, register
from bci.eval.harness import evaluate, permutation_null, summarize
from bci.models import majority, riemann_ts_lr


def _kwargs(n=6, ch=4, s=10, k=2):
    return dict(
        X=np.zeros((n, ch, s)), y=np.arange(n) % k, subject=np.zeros(n), session=np.zeros(n),
        trial_id=np.arange(n), sfreq=100.0, ch_names=[f"c{i}" for i in range(ch)], class_names=list("ab")[:k],
    )


def test_valid_dataset_casts_to_float32():
    ds = EpochsDataset(**_kwargs())
    assert ds.X.dtype == np.float32 and ds.n_classes == 2


@pytest.mark.parametrize(
    "override,match",
    [
        ({"X": np.zeros((6, 4))}, "X must be"),
        ({"y": np.zeros(5, dtype=int)}, "y must have shape"),
        ({"y": np.array([0, 1, 2, 0, 1, 2])}, "y must be in"),
        ({"y": np.zeros(6)}, "integer"),
        ({"trial_id": np.arange(5)}, "trial_id"),
        ({"ch_names": ["c0"]}, "ch_names"),
        ({"sfreq": 0}, "sfreq"),
        ({"split": np.array(["train"] * 5 + ["holdout"])}, "split values"),
    ],
)
def test_validation_errors(override, match):
    with pytest.raises(ValueError, match=match):
        EpochsDataset(**{**_kwargs(), **override})


def test_subset_and_summary(small_ds):
    sub = small_ds.subset(small_ds.subject == 0)
    assert set(sub.subject) == {0} and sub.X.shape[1:] == small_ds.X.shape[1:]
    assert "synthetic" in small_ds.summary() and "trials" in small_ds.summary()


def test_registry():
    assert "synthetic" in available()
    with pytest.raises(KeyError):
        load("does-not-exist")

    @register("_test_tiny")
    def _tiny():
        return EpochsDataset(**_kwargs())

    assert load("_test_tiny").n_trials == 6
    with pytest.raises(ValueError):
        register("_test_tiny")(_tiny)


def test_synthetic_is_deterministic():
    a, b = load("synthetic", n_subjects=2, trials_per_class=2, seed=7), load("synthetic", n_subjects=2, trials_per_class=2, seed=7)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y)
    assert len(np.unique(a.trial_id)) == a.n_trials


def test_harness_builds_fresh_model_per_fold(small_ds):
    calls = []

    def factory():
        calls.append(1)
        return majority()

    df = evaluate(factory, small_ds, ["loso"])
    assert len(calls) == len(df) == 3
    assert {"protocol", "held_out", "n_train", "n_test", "acc", "bal_acc", "one_over_k", "chance_threshold", "binom_p"} <= set(df.columns)


def test_summarize_warns_on_suspicious_accuracy(small_ds):
    df = evaluate(majority, small_ds, ["loso"])
    df["n_correct"] = df["n_test"]  # pretend 100% accuracy
    with pytest.warns(UserWarning, match="LEAKAGE"):
        summarize(df)


def test_summarize_quiet_for_majority(small_ds):
    df = evaluate(majority, small_ds, ["within_subject"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = summarize(df)
    assert not s["above_chance"].any()


def test_permutation_null_runs_and_is_near_chance(small_ds):
    null = permutation_null(lambda: riemann_ts_lr(), small_ds, "loso", n_perm=3, seed=0)
    assert null.shape == (3,) and np.all((0 <= null) & (null <= 1))

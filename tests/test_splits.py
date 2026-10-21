from dataclasses import replace

import numpy as np
import pytest

from bci.eval.splits import cross_session, get_folds, loso, predefined, within_subject


def test_loso_subjects_disjoint_and_cover_all(small_ds):
    folds = list(loso(small_ds))
    assert len(folds) == len(np.unique(small_ds.subject))
    for f in folds:
        assert np.intersect1d(small_ds.subject[f.train_idx], small_ds.subject[f.test_idx]).size == 0
        assert set(small_ds.subject[f.test_idx]) == {f.subject}
    tested = np.concatenate([f.test_idx for f in folds])
    assert np.array_equal(np.sort(tested), np.arange(small_ds.n_trials))


def test_cross_session_sessions_disjoint_within_subject(small_ds):
    folds = list(cross_session(small_ds))
    assert len(folds) == 3 * 2
    for f in folds:
        assert set(small_ds.subject[f.train_idx]) == set(small_ds.subject[f.test_idx]) == {f.subject}
        assert np.intersect1d(small_ds.session[f.train_idx], small_ds.session[f.test_idx]).size == 0
        assert len(np.unique(small_ds.session[f.test_idx])) == 1


def test_cross_session_single_session_raises(small_ds):
    one_session = small_ds.subset(small_ds.session == 0)
    with pytest.raises(ValueError, match="sessions per subject"):
        list(cross_session(one_session))


def test_within_subject_tests_every_trial_exactly_once(small_ds):
    folds = list(within_subject(small_ds, n_splits=4, seed=0))
    tested = np.concatenate([f.test_idx for f in folds])
    assert len(tested) == small_ds.n_trials
    assert np.array_equal(np.sort(tested), np.arange(small_ds.n_trials))
    for f in folds:
        assert set(small_ds.subject[f.train_idx]) == set(small_ds.subject[f.test_idx]) == {f.subject}


def test_within_subject_is_deterministic_given_seed(small_ds):
    a = [f.test_idx for f in within_subject(small_ds, seed=3)]
    b = [f.test_idx for f in within_subject(small_ds, seed=3)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_predefined_uses_split_column(small_ds):
    rng = np.random.default_rng(0)
    split = rng.choice(["train", "val", "test"], size=small_ds.n_trials, p=[0.6, 0.1, 0.3])
    ds = replace(small_ds, split=split)
    for f in predefined(ds):
        assert set(ds.split[f.test_idx]) == {"test"}
        assert set(ds.split[f.train_idx]) <= {"train", "val"}
        assert set(ds.subject[f.train_idx]) == set(ds.subject[f.test_idx]) == {f.subject}


def test_predefined_without_split_raises(small_ds):
    with pytest.raises(ValueError):
        list(predefined(small_ds))


def test_unknown_protocol(small_ds):
    with pytest.raises(KeyError):
        get_folds(small_ds, "random_kfold")

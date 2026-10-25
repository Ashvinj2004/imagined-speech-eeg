"""Cropping (cutting several shorter windows from one trial) is a classic leakage source:
near-identical crops of one trial in both train and test inflate accuracy. Crops inherit their
parent's trial_id, and every splitter must keep all crops of a trial on the same side."""

import numpy as np

from bci.data.base import EpochsDataset
from bci.eval.splits import cross_session, loso, within_subject


def make_crops(ds: EpochsDataset, n_crops: int = 3) -> EpochsDataset:
    """Cut each trial into n_crops consecutive windows; each crop keeps its parent's labels and trial_id."""
    win = ds.X.shape[2] // n_crops
    X = np.concatenate([ds.X[:, :, i * win : (i + 1) * win] for i in range(n_crops)])
    rep = lambda a: np.concatenate([a] * n_crops)  # noqa: E731
    return EpochsDataset(
        X=X, y=rep(ds.y), subject=rep(ds.subject), session=rep(ds.session), trial_id=rep(ds.trial_id),
        sfreq=ds.sfreq, ch_names=ds.ch_names, class_names=ds.class_names, name=ds.name + "-crops",
    )


def _assert_no_straddle(ds, folds):
    for f in folds:
        assert np.intersect1d(ds.trial_id[f.train_idx], ds.trial_id[f.test_idx]).size == 0
        # every crop of every tested trial is in the test set
        for tid in np.unique(ds.trial_id[f.test_idx]):
            assert set(np.flatnonzero(ds.trial_id == tid)) <= set(f.test_idx)


def test_crops_never_straddle_within_subject(small_ds):
    crops = make_crops(small_ds)
    assert crops.n_trials == 3 * small_ds.n_trials
    _assert_no_straddle(crops, list(within_subject(crops, n_splits=4, seed=0)))


def test_crops_never_straddle_other_protocols(small_ds):
    crops = make_crops(small_ds)
    _assert_no_straddle(crops, list(cross_session(crops)))
    _assert_no_straddle(crops, list(loso(crops)))


def test_exact_duplicates_never_straddle(small_ds):
    """Worst case: identical copies of each trial. A naive KFold would put copies on both sides."""
    idx = np.concatenate([np.arange(small_ds.n_trials)] * 2)
    dup = small_ds.subset(idx)
    _assert_no_straddle(dup, list(within_subject(dup, n_splits=4, seed=0)))

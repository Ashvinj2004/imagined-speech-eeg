import numpy as np
import pytest

from bci.data.crops import crop
from bci.eval.harness import _vote_per_trial, evaluate
from bci.eval.splits import loso, within_subject
from bci.models import majority, riemann_ea_lr, riemann_ts_lr


def test_crop_shapes_and_ids(small_ds):
    c = crop(small_ds, window_s=0.25, step_s=0.125)  # 0.5 s trials at 128 Hz -> 3 windows
    assert c.meta["crops"]["n_per_trial"] == 3
    assert c.n_trials == 3 * small_ds.n_trials and c.X.shape[2] == 32
    # crops of one trial are adjacent, share its id and label, and hold the right samples
    assert np.array_equal(c.trial_id[:3], [small_ds.trial_id[0]] * 3)
    assert np.array_equal(c.X[1], small_ds.X[0, :, 16:48])
    assert np.array_equal(c.y, np.repeat(small_ds.y, 3))
    with pytest.raises(ValueError):
        crop(small_ds, window_s=5.0, step_s=0.1)


def test_crops_never_straddle_and_are_counted_once(small_ds):
    c = crop(small_ds, window_s=0.25, step_s=0.125)
    for f in list(within_subject(c, n_splits=4)) + list(loso(c)):
        assert np.intersect1d(c.trial_id[f.train_idx], c.trial_id[f.test_idx]).size == 0
    df, preds = evaluate(majority, c, ["loso"], return_predictions=True)
    assert df["n_test"].sum() == small_ds.n_trials  # trials, not crops
    assert len(preds) == small_ds.n_trials


def test_vote_per_trial_majority_and_passthrough():
    ids = np.array(["a", "a", "a", "b", "b"])
    idx, yt, yp = _vote_per_trial(ids, np.arange(5), np.array([1, 1, 1, 0, 0]), np.array([1, 2, 1, 0, 2]))
    assert idx.tolist() == [0, 3] and yt.tolist() == [1, 0] and yp.tolist() == [1, 0]
    plain = _vote_per_trial(np.array(["a", "b"]), np.arange(2), np.array([0, 1]), np.array([1, 1]))
    assert plain[2].tolist() == [1, 1]


def test_euclidean_alignment_runs_and_records_unseen(small_ds):
    df = evaluate(riemann_ea_lr, small_ds, ["loso", "within_subject"])
    assert set(df["protocol"]) == {"loso", "within_subject"}
    m = riemann_ea_lr().fit(small_ds.X, small_ds.y, subject=small_ds.subject)
    m.predict(small_ds.X[:5], subject=np.full(5, 99))
    assert m.unseen_subjects_ == [99]
    with pytest.raises(ValueError):
        riemann_ea_lr().fit(small_ds.X, small_ds.y)


def test_alignment_whitens_each_subject():
    from pyriemann.estimation import Covariances

    from bci.models.alignment import _whiten

    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4, 100)) * rng.uniform(0.5, 3, size=(1, 4, 1))
    covs = Covariances("oas").fit_transform(X)
    aligned = _whiten(covs, covs.mean(axis=0))
    assert np.allclose(aligned.mean(axis=0), np.eye(4), atol=1e-6)

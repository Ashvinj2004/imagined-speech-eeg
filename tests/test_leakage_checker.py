import numpy as np
import pytest

from bci.eval.splits import Fold, LeakageError, check_no_leakage, cross_session, loso, within_subject


def test_clean_folds_pass(small_ds):
    for gen in (within_subject(small_ds), cross_session(small_ds), loso(small_ds)):
        for f in gen:
            assert check_no_leakage(small_ds, f) is f


def test_injected_trial_leak_is_caught(small_ds):
    f = next(within_subject(small_ds))
    leaked = Fold(f.protocol, f.held_out, f.subject, np.append(f.train_idx, f.test_idx[0]), f.test_idx)
    with pytest.raises(LeakageError):
        check_no_leakage(small_ds, leaked)


def test_injected_crop_leak_is_caught(small_ds):
    """Different row, same trial_id (i.e. a sibling crop) on both sides."""
    dup = small_ds.subset(np.concatenate([np.arange(small_ds.n_trials)] * 2))
    n = small_ds.n_trials
    tr, te = np.arange(0, n), np.arange(n, 2 * n)  # copy A in train, copy B in test
    with pytest.raises(LeakageError, match="trial_id"):
        check_no_leakage(dup, Fold("within_subject", "injected", 0, tr, te))


def test_injected_subject_leak_is_caught_for_loso(small_ds):
    f = next(loso(small_ds))
    # Move one held-out subject's trial into training (row indices still disjoint)
    leaked = Fold("loso", f.held_out, f.subject, np.append(f.train_idx, f.test_idx[0]), f.test_idx[1:])
    with pytest.raises(LeakageError, match="subject"):
        check_no_leakage(small_ds, leaked)


def test_injected_session_leak_is_caught_for_cross_session(small_ds):
    f = next(cross_session(small_ds))
    leaked = Fold("cross_session", f.held_out, f.subject, np.append(f.train_idx, f.test_idx[0]), f.test_idx[1:])
    with pytest.raises(LeakageError, match="session"):
        check_no_leakage(small_ds, leaked)


def test_empty_side_is_caught(small_ds):
    with pytest.raises(LeakageError, match="empty"):
        check_no_leakage(small_ds, Fold("loso", "x", 0, np.arange(10), np.array([], dtype=int)))

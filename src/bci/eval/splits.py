"""Cross-validation splitters. Every fold is checked for leakage before it is yielded.

Protocols:
- within_subject: train and test on the same person (k-fold over that person's trials).
- cross_session:  train on some recording sessions of a person, test on a held-out session.
- loso:           leave-one-subject-out; train on everyone else, test on the held-out person.
- predefined:     use the dataset authors' train/val/test labels (per subject).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from bci.data.base import EpochsDataset


class LeakageError(RuntimeError):
    """Raised when a trial, subject or session appears on both sides of a split."""


@dataclass(frozen=True)
class Fold:
    protocol: str
    held_out: str  # human-readable id of the held-out unit, e.g. "subject=3/fold=2"
    subject: object  # subject whose trials are in the test set
    train_idx: np.ndarray
    test_idx: np.ndarray


def check_no_leakage(ds: EpochsDataset, fold: Fold) -> Fold:
    tr, te = fold.train_idx, fold.test_idx
    if len(tr) == 0 or len(te) == 0:
        raise LeakageError(f"{fold.protocol} {fold.held_out}: empty train or test set")
    if np.intersect1d(tr, te).size:
        raise LeakageError(f"{fold.protocol} {fold.held_out}: same row index in train and test")
    shared = np.intersect1d(ds.trial_id[tr], ds.trial_id[te])
    if shared.size:
        raise LeakageError(
            f"{fold.protocol} {fold.held_out}: {shared.size} trial_id(s) in both train and test "
            f"(e.g. {shared[:3].tolist()}); crops of one trial must stay on one side"
        )
    if fold.protocol == "loso":
        shared = np.intersect1d(ds.subject[tr], ds.subject[te])
        if shared.size:
            raise LeakageError(f"loso {fold.held_out}: subject(s) {shared.tolist()} in both train and test")
    if fold.protocol == "cross_session":
        # Sessions are only meaningful within a subject, so compare (subject, session) pairs.
        pairs_tr = set(zip(ds.subject[tr].tolist(), ds.session[tr].tolist()))
        pairs_te = set(zip(ds.subject[te].tolist(), ds.session[te].tolist()))
        if pairs_tr & pairs_te:
            raise LeakageError(f"cross_session {fold.held_out}: session(s) {sorted(pairs_tr & pairs_te)} on both sides")
    return fold


def within_subject(ds: EpochsDataset, n_splits: int = 5, seed: int = 0) -> Iterator[Fold]:
    for s in np.unique(ds.subject):
        idx = np.flatnonzero(ds.subject == s)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for f, (tr, te) in enumerate(cv.split(idx, ds.y[idx], groups=ds.trial_id[idx])):
            yield check_no_leakage(ds, Fold("within_subject", f"subject={s}/fold={f}", s, idx[tr], idx[te]))


def cross_session(ds: EpochsDataset) -> Iterator[Fold]:
    for s in np.unique(ds.subject):
        idx = np.flatnonzero(ds.subject == s)
        sessions = np.unique(ds.session[idx])
        if len(sessions) < 2:
            raise ValueError(
                f"cross_session needs >= 2 sessions per subject; subject {s} has only {sessions.tolist()}. "
                "Use within_subject for single-session data."
            )
        for ses in sessions:
            te = idx[ds.session[idx] == ses]
            tr = idx[ds.session[idx] != ses]
            yield check_no_leakage(ds, Fold("cross_session", f"subject={s}/session={ses}", s, tr, te))


def loso(ds: EpochsDataset) -> Iterator[Fold]:
    subjects = np.unique(ds.subject)
    if len(subjects) < 2:
        raise ValueError("loso needs >= 2 subjects")
    for s in subjects:
        te = np.flatnonzero(ds.subject == s)
        tr = np.flatnonzero(ds.subject != s)
        yield check_no_leakage(ds, Fold("loso", f"subject={s}", s, tr, te))


def predefined(ds: EpochsDataset) -> Iterator[Fold]:
    """Per subject: train on split in {train, val}, test on split == test (val is for in-pipeline early stopping)."""
    if ds.split is None:
        raise ValueError("predefined protocol needs ds.split")
    for s in np.unique(ds.subject):
        m = ds.subject == s
        tr = np.flatnonzero(m & np.isin(ds.split, ["train", "val"]))
        te = np.flatnonzero(m & (ds.split == "test"))
        yield check_no_leakage(ds, Fold("predefined", f"subject={s}", s, tr, te))


PROTOCOLS = {
    "within_subject": within_subject,
    "cross_session": cross_session,
    "loso": loso,
    "predefined": predefined,
}


def get_folds(ds: EpochsDataset, protocol: str, seed: int = 0) -> Iterator[Fold]:
    if protocol not in PROTOCOLS:
        raise KeyError(f"unknown protocol {protocol!r}; choose from {sorted(PROTOCOLS)}")
    if protocol == "within_subject":
        return within_subject(ds, seed=seed)
    return PROTOCOLS[protocol](ds)

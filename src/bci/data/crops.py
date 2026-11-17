"""Cropped training: cut each trial into several shorter, overlapping windows.

Each crop keeps its parent's label, subject, session and, crucially, `trial_id`, so every splitter
keeps all crops of one trial on the same side of a split (RULES.md rule 1). At test time the
harness combines the predictions of a trial's crops by majority vote and counts the trial once,
so chance thresholds are computed on real trials, not on correlated crops.
"""

from __future__ import annotations

import numpy as np

from bci.data.base import EpochsDataset


def crop(ds: EpochsDataset, window_s: float, step_s: float) -> EpochsDataset:
    """Sliding windows of `window_s` seconds every `step_s` seconds, within each trial."""
    win = int(round(window_s * ds.sfreq))
    step = int(round(step_s * ds.sfreq))
    n_samp = ds.X.shape[2]
    if not 0 < win <= n_samp or step <= 0:
        raise ValueError(f"window {win} samples must be in 1..{n_samp} and step {step} positive")
    starts = list(range(0, n_samp - win + 1, step))
    idx = np.repeat(np.arange(ds.n_trials), len(starts))
    X = np.concatenate([ds.X[:, :, s : s + win] for s in starts], axis=0)
    # concatenation above orders crops window-major; reorder so each trial's crops sit together
    order = np.argsort(np.tile(np.arange(ds.n_trials), len(starts)), kind="stable")
    X = X[order]
    out = ds.subset(idx)  # repeats every per-trial field, trial_id included
    out.X = X
    out.name = f"{ds.name}-crops{len(starts)}"
    out.meta = {**ds.meta, "crops": {"window_s": window_s, "step_s": step_s, "n_per_trial": len(starts)}}
    return out

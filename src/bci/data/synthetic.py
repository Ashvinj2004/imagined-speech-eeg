"""Synthetic EEG-like dataset for plumbing tests. NEVER report numbers from it.

Generative model (per trial):
    x(t) = A_subject @ (I + P_session) @ s(t) + sensor_noise
- s(t): latent "sources" (brain-signal generators), white noise whose per-source variance
  depends weakly on the class. Classes therefore differ only in *spatial covariance*
  (how channels co-vary), which is what Riemannian models exploit.
- A_subject: random channel mixing per subject (stands in for different head anatomy and
  electrode placement), so a model trained on other subjects transfers poorly (LOSO drop).
- P_session: smaller perturbation per session (stands in for re-placing the EEG cap on
  another day), so held-out sessions score lower than within-session folds.
- Per-trial variance jitter keeps the problem hard (~60-70% within-subject on 5 classes).
"""

from __future__ import annotations

import numpy as np

from bci.data.base import EpochsDataset, register


@register("synthetic")
def make_synthetic(
    n_subjects: int = 6,
    n_sessions: int = 2,
    n_classes: int = 5,
    trials_per_class: int = 30,  # per subject per session
    n_channels: int = 16,
    sfreq: float = 128.0,
    duration: float = 2.0,
    class_sep: float = 0.25,  # std of class-specific log-variance offsets (effect size)
    trial_jitter: float = 0.5,  # std of per-trial log-variance noise (difficulty)
    session_shift: float = 0.35,  # size of per-session mixing perturbation
    noise_std: float = 0.5,  # additive sensor noise
    seed: int = 0,
) -> EpochsDataset:
    rng = np.random.default_rng(seed)
    n_samples = int(round(sfreq * duration))
    class_logvar = class_sep * rng.standard_normal((n_classes, n_channels))  # shared across subjects

    X, y, subj, sess, tid = [], [], [], [], []
    for s in range(n_subjects):
        A_subj = rng.standard_normal((n_channels, n_channels)) / np.sqrt(n_channels)
        for ses in range(n_sessions):
            P = session_shift * rng.standard_normal((n_channels, n_channels)) / np.sqrt(n_channels)
            A = A_subj @ (np.eye(n_channels) + P)
            labels = rng.permutation(np.repeat(np.arange(n_classes), trials_per_class))
            for t, k in enumerate(labels):
                logvar = class_logvar[k] + trial_jitter * rng.standard_normal(n_channels)
                sources = np.exp(logvar / 2)[:, None] * rng.standard_normal((n_channels, n_samples))
                x = A @ sources + noise_std * rng.standard_normal((n_channels, n_samples))
                X.append(x)
                y.append(k)
                subj.append(s)
                sess.append(ses)
                tid.append(f"s{s}-ses{ses}-t{t:03d}")

    return EpochsDataset(
        X=np.stack(X).astype(np.float32),
        y=np.array(y),
        subject=np.array(subj),
        session=np.array(sess),
        trial_id=np.array(tid),
        sfreq=sfreq,
        ch_names=[f"EEG{i:02d}" for i in range(n_channels)],
        class_names=[f"word{i}" for i in range(n_classes)],
        name="synthetic",
        meta={"seed": seed, "synthetic": True},
    )

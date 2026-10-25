"""The data contract every dataset loader must return, plus the loader registry.

An *epoch* (a.k.a. trial) is a fixed-length window of EEG cut around one stimulus/cue,
so a dataset is a 3-D array: (trials, channels, samples).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

VALID_SPLITS = {"train", "val", "test"}


@dataclass
class EpochsDataset:
    X: np.ndarray  # (n_trials, n_channels, n_samples) float32; a "channel" is one electrode's signal
    y: np.ndarray  # (n_trials,) int labels 0..k-1
    subject: np.ndarray  # (n_trials,) subject id per trial
    session: np.ndarray  # (n_trials,) recording session id per trial (cap removed/replaced between sessions)
    trial_id: np.ndarray  # (n_trials,) unique per ORIGINAL trial; crops inherit their parent's id
    sfreq: float  # sampling frequency in Hz (samples per second)
    ch_names: list[str]  # electrode names, e.g. "Cz" (10-20 system positions)
    class_names: list[str]  # class_names[i] is the word/label for y == i
    name: str = "unnamed"
    split: np.ndarray | None = None  # optional per-trial "train"/"val"/"test" from the dataset authors
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float32)
        if self.X.ndim != 3:
            raise ValueError(f"X must be (trials, channels, samples); got shape {self.X.shape}")
        n, n_ch, _ = self.X.shape

        self.y = np.asarray(self.y)
        if not np.issubdtype(self.y.dtype, np.integer):
            raise ValueError(f"y must be integer labels; got dtype {self.y.dtype}")
        self.y = self.y.astype(np.int64)

        self.subject = np.asarray(self.subject)
        self.session = np.asarray(self.session)
        self.trial_id = np.asarray(self.trial_id)
        per_trial = {"y": self.y, "subject": self.subject, "session": self.session, "trial_id": self.trial_id}
        if self.split is not None:
            self.split = np.asarray(self.split).astype(str)
            per_trial["split"] = self.split
        for key, arr in per_trial.items():
            if arr.shape != (n,):
                raise ValueError(f"{key} must have shape ({n},) to match X; got {arr.shape}")

        k = len(self.class_names)
        if k < 2:
            raise ValueError("need at least 2 classes")
        if n and (self.y.min() < 0 or self.y.max() >= k):
            raise ValueError(f"y must be in 0..{k - 1} to match class_names; got range {self.y.min()}..{self.y.max()}")
        if len(self.ch_names) != n_ch:
            raise ValueError(f"ch_names has {len(self.ch_names)} entries but X has {n_ch} channels")
        if self.sfreq <= 0:
            raise ValueError(f"sfreq must be positive; got {self.sfreq}")
        if self.split is not None:
            bad = set(np.unique(self.split)) - VALID_SPLITS
            if bad:
                raise ValueError(f"split values must be in {sorted(VALID_SPLITS)}; got {sorted(bad)}")

    @property
    def n_trials(self) -> int:
        return self.X.shape[0]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def subset(self, idx) -> "EpochsDataset":
        """Return a new, re-validated dataset with only the selected trials (bool mask or int indices)."""
        idx = np.asarray(idx)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return replace(
            self,
            X=self.X[idx],
            y=self.y[idx],
            subject=self.subject[idx],
            session=self.session[idx],
            trial_id=self.trial_id[idx],
            split=None if self.split is None else self.split[idx],
            ch_names=list(self.ch_names),
            class_names=list(self.class_names),
            meta=dict(self.meta),
        )

    def summary(self) -> str:
        n, n_ch, n_s = self.X.shape
        lines = [
            f"{self.name}: {n} trials x {n_ch} channels x {n_s} samples "
            f"({n_s / self.sfreq:.2f} s at {self.sfreq:g} Hz)",
            f"  classes ({self.n_classes}): "
            + ", ".join(f"{c}={int((self.y == i).sum())}" for i, c in enumerate(self.class_names)),
            f"  subjects: {len(np.unique(self.subject))}, sessions per subject: "
            + ", ".join(
                f"{s}:{len(np.unique(self.session[self.subject == s]))}" for s in np.unique(self.subject)
            ),
            f"  unique original trials: {len(np.unique(self.trial_id))}",
        ]
        if self.split is not None:
            lines.append("  split: " + ", ".join(f"{s}={int((self.split == s).sum())}" for s in np.unique(self.split)))
        return "\n".join(lines)


_REGISTRY: dict[str, Callable[..., EpochsDataset]] = {}


def register(name: str):
    """Decorator: register a loader function under `name` so `load(name, **kwargs)` can find it."""

    def deco(fn: Callable[..., EpochsDataset]):
        if name in _REGISTRY:
            raise ValueError(f"dataset {name!r} is already registered")
        _REGISTRY[name] = fn
        return fn

    return deco


def load(name: str, **kwargs) -> EpochsDataset:
    if name not in _REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; available: {available()}")
    ds = _REGISTRY[name](**kwargs)
    if not isinstance(ds, EpochsDataset):
        raise TypeError(f"loader {name!r} must return an EpochsDataset, got {type(ds).__name__}")
    return ds


def available() -> list[str]:
    return sorted(_REGISTRY)

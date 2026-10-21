"""Nieto et al. 2022 "Thinking out loud": inner speech, 10 subjects x 3 sessions, 4 Spanish directions.

Source: OpenNeuro ds003626 (download with `python scripts/download_data.py nieto`). We use the
authors' preprocessed epochs (derivatives/sub-XX/ses-0Y/*_eeg-epo.fif): 128 EEG channels,
re-referenced, 50 Hz notch (removes mains-power interference), 0.5-100 Hz band-pass, decimated
to 256 Hz, epochs from -0.5 s to 4 s around the visual cue.

Each session mixes three conditions: pronounced (said aloud), inner (imagined, our target) and
visualized. The per-trial condition only exists in *_events.dat, a pickled numpy array with
columns [sample, class 0-3, condition 0-2, session]. Unpickling a downloaded file can run
arbitrary code, so `_read_events` only allows the handful of numpy classes a plain array needs.
"""

from __future__ import annotations

import importlib
import pickle
from pathlib import Path

import numpy as np

from bci.data.base import EpochsDataset, register

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "nieto2022"
CLASS_NAMES = ["Arriba/Up", "Abajo/Down", "Derecha/Right", "Izquierda/Left"]
CONDITIONS = {"pronounced": 0, "inner": 1, "visualized": 2}
EVENT_CODES = [31, 32, 33, 34]  # trigger codes for the 4 classes in the .fif events

_ALLOWED_PICKLE = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("_codecs", "encode"),
}


class _ArrayOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOWED_PICKLE:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name} from a downloaded file")
        if module == "numpy.core.multiarray":  # renamed in numpy 2
            module = "numpy._core.multiarray"
        return getattr(importlib.import_module(module), name)


def _read_events(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        events = _ArrayOnlyUnpickler(f).load()
    if not isinstance(events, np.ndarray) or events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"{path}: expected an (n_trials, 4) array, got {type(events).__name__} {getattr(events, 'shape', '')}")
    return events.astype(np.int64)


@register("nieto")
def load_nieto(
    root: str | Path = DEFAULT_ROOT,
    subjects=None,
    sessions=(1, 2, 3),
    condition: str = "inner",
    tmin: float = 1.0,  # authors' recommended "action interval": 1.0-3.5 s after the cue
    tmax: float = 3.5,
) -> EpochsDataset:
    import mne

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} not found; run `python scripts/download_data.py nieto` first")
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(CONDITIONS)}")
    subjects = subjects or range(1, 11)

    X, y, subj, sess, tid = [], [], [], [], []
    sfreq = ch_names = None
    for s in subjects:
        for b in sessions:
            stem = root / "derivatives" / f"sub-{s:02d}" / f"ses-{b:02d}" / f"sub-{s:02d}_ses-{b:02d}"
            events = _read_events(Path(f"{stem}_events.dat"))
            epochs = mne.read_epochs(f"{stem}_eeg-epo.fif", preload=True, verbose="ERROR")
            if len(epochs) != len(events):
                raise ValueError(f"{stem}: {len(epochs)} epochs but {len(events)} event rows")
            if not np.array_equal(np.asarray(EVENT_CODES)[events[:, 1]], epochs.events[:, 2]):
                raise ValueError(f"{stem}: class labels in events.dat disagree with the .fif trigger codes")
            epochs.crop(tmin=tmin, tmax=tmax, include_tmax=False)
            if sfreq is None:
                sfreq, ch_names = epochs.info["sfreq"], list(epochs.ch_names)
            elif epochs.info["sfreq"] != sfreq or list(epochs.ch_names) != ch_names:
                raise ValueError(f"{stem}: sampling rate or channels differ from earlier files")
            keep = np.flatnonzero(events[:, 2] == CONDITIONS[condition])
            data = epochs.get_data(copy=False) * 1e6  # Volts -> microvolts
            X.append(data[keep])
            y.append(events[keep, 1])
            subj += [s] * len(keep)
            sess += [b] * len(keep)
            # Index within the session's full trial list, so ids stay stable whatever condition is picked
            tid += [f"nieto-s{s:02d}-ses{b}-t{i:03d}" for i in keep]

    return EpochsDataset(
        X=np.concatenate(X).astype(np.float32),
        y=np.concatenate(y),
        subject=np.array(subj),
        session=np.array(sess),
        trial_id=np.array(tid),
        sfreq=float(sfreq),
        ch_names=ch_names,
        class_names=list(CLASS_NAMES),
        name=f"nieto2022-{condition}",
        meta={"source": "OpenNeuro ds003626 derivatives", "condition": condition, "window_s": (tmin, tmax), "units": "uV"},
    )

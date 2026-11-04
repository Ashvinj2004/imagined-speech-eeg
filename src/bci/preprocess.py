"""Fixed, data-independent preprocessing. Safe to run before splitting (RULES.md rule 1).

Nothing here learns from the data: every step is a fixed mathematical operation applied to each
trial on its own. Anything that learns (z-scoring, ICA, alignment...) belongs inside a model
pipeline instead, so the harness fits it on the training fold only.

Steps, in order:
1. Re-reference: express each channel relative to the average of all channels at that instant
   ("common average reference"), removing activity shared by every electrode.
2. Notch filter: remove one frequency, the mains-power hum (60 Hz in Korea, 50 Hz in Europe/Argentina).
3. Band-pass filter: keep only frequencies between l_freq and h_freq, removing slow drift and
   fast muscle noise.
4. Resample: change the sampling rate (measurements per second), e.g. to make datasets match.

Every step works along the time axis of each trial separately, so one trial's values never depend
on another trial. This matters for Track 3, whose public copy stores trials glued end to end:
filtering the glued recording would smear neighbouring trials into each other.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import gcd

import numpy as np
from scipy import signal

from bci.data.base import EpochsDataset


@dataclass(frozen=True)
class PreprocessConfig:
    reref: str | None = "average"  # "average" or None
    notch_hz: float | None = None  # mains frequency to remove; None skips the notch
    l_freq: float | None = 0.5  # band-pass low edge in Hz; None = no high-pass
    h_freq: float | None = 40.0  # band-pass high edge in Hz; None = no low-pass
    filter_order: int = 4  # Butterworth order; applied forward and backward, so no time shift
    notch_q: float = 30.0  # notch sharpness: removed band is about notch_hz / notch_q wide
    resample_hz: float | None = None  # target sampling rate; None keeps the original


# Per-dataset defaults. Track 3 was recorded in Korea (60 Hz mains). Nieto's published epochs are
# already re-referenced, notch-filtered at 50 Hz and band-passed 0.5-100 Hz by the authors.
DEFAULTS = {
    "bci2020_track3": PreprocessConfig(notch_hz=60.0),
    "nieto2022": PreprocessConfig(reref=None),
}


def default_config(ds: EpochsDataset) -> PreprocessConfig:
    key = next((k for k in DEFAULTS if ds.name.startswith(k)), None)
    return DEFAULTS[key] if key else PreprocessConfig()


def preprocess(ds: EpochsDataset, cfg: PreprocessConfig | None = None) -> EpochsDataset:
    """Return a new dataset with the fixed preprocessing applied; `ds` is left unchanged."""
    cfg = cfg or default_config(ds)
    fs = float(ds.sfreq)
    nyq = fs / 2
    for name, f in (("notch_hz", cfg.notch_hz), ("l_freq", cfg.l_freq), ("h_freq", cfg.h_freq)):
        if f is not None and not 0 < f < nyq:
            raise ValueError(f"{name}={f} Hz must be between 0 and the Nyquist frequency ({nyq} Hz)")
    if cfg.l_freq is not None and cfg.h_freq is not None and cfg.l_freq >= cfg.h_freq:
        raise ValueError(f"l_freq ({cfg.l_freq}) must be below h_freq ({cfg.h_freq})")
    if cfg.reref not in (None, "average"):
        raise ValueError(f"unknown reref {cfg.reref!r}; use 'average' or None")

    X = ds.X.astype(np.float64)
    # padlen: how much mirrored signal is added at each end so filters don't distort trial edges
    padlen = min(X.shape[-1] - 1, int(3 * fs))

    if cfg.reref == "average":
        X = X - X.mean(axis=1, keepdims=True)

    if cfg.notch_hz is not None:
        b, a = signal.iirnotch(cfg.notch_hz, cfg.notch_q, fs=fs)
        X = signal.filtfilt(b, a, X, axis=-1, padtype="odd", padlen=padlen)

    if cfg.l_freq is not None or cfg.h_freq is not None:
        if cfg.l_freq is not None and cfg.h_freq is not None:
            sos = signal.butter(cfg.filter_order, [cfg.l_freq, cfg.h_freq], btype="bandpass", fs=fs, output="sos")
        elif cfg.l_freq is not None:
            sos = signal.butter(cfg.filter_order, cfg.l_freq, btype="highpass", fs=fs, output="sos")
        else:
            sos = signal.butter(cfg.filter_order, cfg.h_freq, btype="lowpass", fs=fs, output="sos")
        X = signal.sosfiltfilt(sos, X, axis=-1, padtype="odd", padlen=padlen)

    if cfg.resample_hz is not None and cfg.resample_hz != fs:
        if cfg.h_freq is None or cfg.h_freq >= cfg.resample_hz / 2:
            raise ValueError("set h_freq below half the new sampling rate before resampling (avoids aliasing)")
        up, down = int(round(cfg.resample_hz)), int(round(fs))
        g = gcd(up, down)
        X = signal.resample_poly(X, up // g, down // g, axis=-1, padtype="line")
        fs = float(cfg.resample_hz)

    meta = dict(ds.meta)
    meta["preprocess"] = asdict(cfg)
    return replace(ds, X=X.astype(np.float32), sfreq=fs, meta=meta)

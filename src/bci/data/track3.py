"""BCI Competition 2020 Track 3: imagined speech, 15 subjects, 5 words/phrases, 64 channels at 256 Hz.

Source: NEMAR nm000113, a BIDS conversion of https://osf.io/pq7vb/ (download with
`python scripts/download_data.py track3`). Each subject has three BIDS "runs" that are the
competition's predefined splits: run-00 = training (60 trials/class), run-01 = validation
(10/class), run-02 = test (10/class, labels from the organisers' answer sheet).

The converter concatenated the original epochs into one continuous recording per run and
marks each trial with an annotation (onset + duration), so we cut the epochs back out here.
All trials were recorded in a single session, so cross-session evaluation is not possible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bci.data.base import EpochsDataset, register

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "bci2020_track3"
CLASS_NAMES = ["Hello", "Help me", "Stop", "Thank you", "Yes"]
RUN_TO_SPLIT = {"00": "train", "01": "val", "02": "test"}


@register("track3")
def load_track3(root: str | Path = DEFAULT_ROOT, subjects=None) -> EpochsDataset:
    import mne

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} not found; run `python scripts/download_data.py track3` first")
    subjects = subjects or range(1, 16)

    X, y, subj, split, tid = [], [], [], [], []
    sfreq = ch_names = None
    for s in subjects:
        eeg_dir = root / f"sub-{s:02d}" / "eeg"
        for run, split_name in RUN_TO_SPLIT.items():
            stem = eeg_dir / f"sub-{s:02d}_task-imaginedSpeech_run-{run}"
            raw = mne.io.read_raw_edf(f"{stem}_eeg.edf", preload=True, verbose="ERROR")
            events = pd.read_csv(f"{stem}_events.tsv", sep="\t")
            fs = raw.info["sfreq"]
            if sfreq is None:
                sfreq, ch_names = fs, list(raw.ch_names)
            elif fs != sfreq or list(raw.ch_names) != ch_names:
                raise ValueError(f"{stem}: sampling rate or channels differ from earlier files")
            data = raw.get_data() * 1e6  # Volts -> microvolts (EEG amplitudes are tens of µV)
            n_samp = int(round(events["duration"].iloc[0] * fs))
            for i, ev in events.reset_index(drop=True).iterrows():
                start = int(round(ev["onset"] * fs))
                if start + n_samp > data.shape[1]:
                    raise ValueError(f"{stem}: trial {i} runs past the end of the recording")
                X.append(data[:, start : start + n_samp])
                y.append(CLASS_NAMES.index(ev["trial_type"]))
                subj.append(s)
                split.append(split_name)
                tid.append(f"track3-s{s:02d}-run{run}-t{i:03d}")

    return EpochsDataset(
        X=np.stack(X).astype(np.float32),
        y=np.array(y),
        subject=np.array(subj),
        session=np.zeros(len(y), dtype=int),
        trial_id=np.array(tid),
        split=np.array(split),
        sfreq=float(sfreq),
        ch_names=ch_names,
        class_names=list(CLASS_NAMES),
        name="bci2020_track3",
        meta={"source": "NEMAR nm000113 (BCI Competition 2020 Track 3)", "units": "uV"},
    )

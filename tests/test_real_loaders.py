"""Loader checks for the real datasets. The data-dependent tests skip when data/ is not downloaded
(e.g. in CI); run `python scripts/download_data.py` locally to enable them."""

import io
import pickle

import numpy as np
import pytest

from bci.data import load
from bci.data.nieto import DEFAULT_ROOT as NIETO_ROOT
from bci.data.nieto import _ArrayOnlyUnpickler
from bci.data.track3 import DEFAULT_ROOT as TRACK3_ROOT
from bci.eval.splits import loso, predefined


class _Evil:
    def __reduce__(self):
        return (print, ("this must never run",))


def test_safe_unpickler_reads_plain_arrays():
    arr = np.arange(12).reshape(3, 4)
    assert np.array_equal(_ArrayOnlyUnpickler(io.BytesIO(pickle.dumps(arr, protocol=2))).load(), arr)


def test_safe_unpickler_refuses_code(capsys):
    with pytest.raises(pickle.UnpicklingError, match="refusing"):
        _ArrayOnlyUnpickler(io.BytesIO(pickle.dumps(_Evil()))).load()
    assert "must never run" not in capsys.readouterr().out


track3_present = pytest.mark.skipif(not (TRACK3_ROOT / "sub-02").exists(), reason="Track 3 not downloaded")
nieto_present = pytest.mark.skipif(
    not (NIETO_ROOT / "derivatives/sub-01/ses-01/sub-01_ses-01_eeg-epo.fif").exists(), reason="Nieto not downloaded"
)


@track3_present
def test_track3_shape_and_splits():
    ds = load("track3", subjects=[1, 2])
    assert ds.X.shape[1:] == (64, ds.X.shape[2]) and ds.sfreq == 256.0
    for s in (1, 2):
        m = ds.subject == s
        counts = {k: np.bincount(ds.y[m & (ds.split == k)], minlength=5).tolist() for k in ("train", "val", "test")}
        assert counts == {"train": [60] * 5, "val": [10] * 5, "test": [10] * 5}
    assert len(np.unique(ds.trial_id)) == ds.n_trials
    assert 1 < np.std(ds.X) < 1000  # microvolts, not volts
    assert len(list(predefined(ds))) == 2 and len(list(loso(ds))) == 2


@nieto_present
def test_nieto_inner_speech_subject1():
    ds = load("nieto", subjects=[1], sessions=(1,))
    assert ds.n_trials == 80 and ds.X.shape[1] == 128 and ds.sfreq == 256.0
    assert np.bincount(ds.y).tolist() == [20, 20, 20, 20]
    assert ds.X.shape[2] == int(2.5 * 256)  # 1.0-3.5 s window
    assert len(np.unique(ds.trial_id)) == ds.n_trials

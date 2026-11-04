import numpy as np
import pytest
from scipy import signal

from bci.data.base import EpochsDataset
from bci.preprocess import PreprocessConfig, default_config, preprocess

FS = 256.0


def _ds(X, name="test"):
    n, ch, _ = X.shape
    return EpochsDataset(
        X=X, y=np.arange(n) % 2, subject=np.zeros(n), session=np.zeros(n), trial_id=np.arange(n),
        sfreq=FS, ch_names=[f"c{i}" for i in range(ch)], class_names=["a", "b"], name=name,
    )


def _sines(freqs, n_trials=4, n_ch=3, seconds=3.0, seed=0):
    t = np.arange(int(seconds * FS)) / FS
    rng = np.random.default_rng(seed)
    X = np.zeros((n_trials, n_ch, t.size))
    for f in freqs:
        X += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi, size=(n_trials, n_ch, 1)))
    return X


def _power_db(X, f):
    """Mean power (dB) at frequency f, over all trials and channels, ignoring the first/last 0.5 s."""
    edge = int(0.5 * FS)
    freqs, psd = signal.welch(X[..., edge:-edge], fs=FS, nperseg=256, axis=-1)
    return 10 * np.log10(psd[..., np.argmin(abs(freqs - f))].mean())


NO_REREF = dict(reref=None)


def test_notch_removes_mains_and_keeps_alpha():
    X = _sines([10, 60])
    out = preprocess(_ds(X), PreprocessConfig(notch_hz=60, l_freq=None, h_freq=None, **NO_REREF)).X
    assert _power_db(X, 60) - _power_db(out, 60) > 20  # mains hum strongly reduced
    assert abs(_power_db(X, 10) - _power_db(out, 10)) < 1  # 10 Hz brain rhythm untouched


def test_bandpass_removes_out_of_band():
    X = _sines([0.1, 10, 80])
    out = preprocess(_ds(X), PreprocessConfig(l_freq=0.5, h_freq=40, **NO_REREF)).X
    assert _power_db(X, 80) - _power_db(out, 80) > 40
    assert abs(_power_db(X, 10) - _power_db(out, 10)) < 1
    # slow drift: compare variance, since 0.1 Hz is below Welch's resolution
    drift = _sines([0.1])
    assert preprocess(_ds(drift), PreprocessConfig(l_freq=0.5, h_freq=40, **NO_REREF)).X.std() < 0.2 * drift.std()


def test_average_reference_zeroes_channel_mean():
    X = _sines([10]) + 5.0  # a shared offset on every channel
    out = preprocess(_ds(X), PreprocessConfig(l_freq=None, h_freq=None)).X
    assert np.allclose(out.mean(axis=1), 0, atol=1e-5)


def test_trials_are_processed_independently():
    """Changing one trial must not change any other: no smearing between neighbouring trials."""
    X = _sines([10, 60], n_trials=6)
    cfg = PreprocessConfig(notch_hz=60, l_freq=0.5, h_freq=40, resample_hz=128)
    a = preprocess(_ds(X), cfg).X
    X2 = X.copy()
    X2[3] += 100 * np.random.default_rng(1).standard_normal(X2[3].shape)
    b = preprocess(_ds(X2), cfg).X
    others = [i for i in range(6) if i != 3]
    assert np.array_equal(a[others], b[others])
    assert not np.allclose(a[3], b[3])


def test_shape_metadata_and_determinism():
    ds = _ds(_sines([10, 60]))
    cfg = PreprocessConfig(notch_hz=60)
    out1, out2 = preprocess(ds, cfg), preprocess(ds, cfg)
    assert out1.X.shape == ds.X.shape and out1.X.dtype == np.float32
    assert np.array_equal(out1.X, out2.X)
    assert np.array_equal(out1.trial_id, ds.trial_id) and np.array_equal(out1.y, ds.y)
    assert out1.meta["preprocess"]["notch_hz"] == 60
    assert not np.shares_memory(out1.X, ds.X)  # input left untouched


def test_resample_changes_rate_and_length():
    ds = _ds(_sines([10], seconds=2.0))
    out = preprocess(ds, PreprocessConfig(h_freq=40, resample_hz=128, **NO_REREF))
    assert out.sfreq == 128.0 and out.X.shape[-1] == 256
    assert abs(_power_db(ds.X, 10) - 10 * np.log10(
        signal.welch(out.X[..., 64:-64], fs=128, nperseg=128, axis=-1)[1][..., 10].mean())) < 1.5


@pytest.mark.parametrize(
    "cfg",
    [
        PreprocessConfig(notch_hz=200),  # above Nyquist (128 Hz)
        PreprocessConfig(l_freq=40, h_freq=10),
        PreprocessConfig(reref="mastoids"),
        PreprocessConfig(h_freq=None, resample_hz=64),  # would alias
    ],
)
def test_invalid_configs_rejected(cfg):
    with pytest.raises(ValueError):
        preprocess(_ds(_sines([10])), cfg)


def test_dataset_defaults():
    assert default_config(_ds(_sines([10]), name="bci2020_track3")).notch_hz == 60.0
    assert default_config(_ds(_sines([10]), name="nieto2022-inner")).reref is None
    assert default_config(_ds(_sines([10]), name="synthetic")) == PreprocessConfig()

import pytest

from bci.data import load


@pytest.fixture(scope="session")
def small_ds():
    """Small synthetic dataset: 3 subjects x 2 sessions x 3 classes x 8 trials."""
    return load("synthetic", n_subjects=3, n_sessions=2, n_classes=3, trials_per_class=8, n_channels=8, duration=0.5, seed=1)

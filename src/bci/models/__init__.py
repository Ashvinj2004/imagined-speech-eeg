"""Model FACTORIES: zero-arg callables returning a fresh, unfitted sklearn-compatible estimator.

Anything that learns from data must live inside the returned estimator so the harness fits it
on the training fold only.
"""

from bci.models.baselines import majority, riemann_ts_lr


def eegnet(**kwargs):
    from bci.models.deep import eegnet as _eegnet  # imported lazily: torch is slow to import

    return _eegnet(**kwargs)


def shallow_convnet(**kwargs):
    from bci.models.deep import shallow_convnet as _shallow

    return _shallow(**kwargs)


__all__ = ["majority", "riemann_ts_lr", "eegnet", "shallow_convnet"]

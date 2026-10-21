"""Model FACTORIES: zero-arg callables returning a fresh, unfitted sklearn-compatible estimator.

Anything that learns from data must live inside the returned estimator so the harness fits it
on the training fold only.
"""

from bci.models.baselines import majority, riemann_ts_lr

__all__ = ["majority", "riemann_ts_lr"]

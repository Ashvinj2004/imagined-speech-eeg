from __future__ import annotations

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def majority() -> DummyClassifier:
    """Always predicts the most frequent training class. Must never beat the chance threshold."""
    return DummyClassifier(strategy="most_frequent")


def riemann_ts_lr(C: float = 1.0, seed: int = 0) -> Pipeline:
    """Classic EEG baseline: per-trial spatial covariance -> tangent space -> logistic regression."""
    return Pipeline(
        [
            # Spatial covariance matrix per trial (channels x channels): how electrodes co-vary.
            # "oas" = Oracle Approximating Shrinkage, a regularised estimate that stays well-conditioned.
            ("cov", Covariances(estimator="oas")),
            # Covariance matrices live on a curved manifold (symmetric positive-definite matrices);
            # the tangent space flattens them around the training-fold mean so a linear model can use them.
            ("ts", TangentSpace(metric="riemann")),
            ("lr", LogisticRegression(C=C, max_iter=1000, random_state=seed)),
        ]
    )

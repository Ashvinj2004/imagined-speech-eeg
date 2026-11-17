"""Euclidean Alignment (He & Wu, 2020): make different people's EEG look more alike.

For each subject, compute the mean of their trials' spatial covariance matrices, R, and whiten
every trial's covariance as R^{-1/2} C R^{-1/2}. After this, each subject's mean covariance is
the identity, so a model trained on several people transfers better to a new one.

Where the reference R comes from (this matters for leakage, RULES.md rule 1):
- `fit` computes R per subject from the TRAINING trials only, and the downstream model is
  trained on the whitened training covariances.
- `predict` on a subject seen in training reuses that subject's training-fold R.
- `predict` on an unseen subject (LOSO) computes R from that subject's own test trials, using
  no labels. This is the standard EA protocol: it is unsupervised calibration on the new person,
  and results that use it must say so. `self.unseen_subjects_` records when it happened.
"""

from __future__ import annotations

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.linalg import fractional_matrix_power
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def _whiten(covs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    w = np.real(fractional_matrix_power(ref, -0.5))
    return np.einsum("ij,njk,kl->nil", w, covs, w)


class EuclideanAlignedClassifier(ClassifierMixin, BaseEstimator):
    """Per-subject whitening of covariances, then a classifier on the aligned matrices.

    The harness passes `subject` to fit/predict because `needs_subject = True`."""

    needs_subject = True

    def __init__(self, C: float = 1.0, seed: int = 0, estimator: str = "oas"):
        self.C = C
        self.seed = seed
        self.estimator = estimator

    def _refs(self, covs: np.ndarray, subject: np.ndarray) -> dict:
        return {s: covs[subject == s].mean(axis=0) for s in np.unique(subject)}

    def _align(self, covs: np.ndarray, subject: np.ndarray, refs: dict) -> np.ndarray:
        out = np.empty_like(covs)
        for s in np.unique(subject):
            m = subject == s
            out[m] = _whiten(covs[m], refs[s])
        return out

    def _covs(self, X) -> np.ndarray:
        # Covariances in chunks of trials: a float64 copy of a whole dataset does not fit on a laptop.
        return np.concatenate([self.cov_.transform(np.asarray(X[i : i + 256], dtype=np.float64))
                               for i in range(0, len(X), 256)])

    def fit(self, X, y, subject=None):
        if subject is None:
            raise ValueError("EuclideanAlignedClassifier needs per-trial `subject` ids")
        subject = np.asarray(subject)
        self.cov_ = Covariances(estimator=self.estimator)
        covs = self._covs(X)
        self.refs_ = self._refs(covs, subject)
        self.clf_ = Pipeline([("ts", TangentSpace(metric="riemann")),
                              ("lr", LogisticRegression(C=self.C, max_iter=1000, random_state=self.seed))])
        self.clf_.fit(self._align(covs, subject, self.refs_), y)
        self.classes_ = self.clf_.classes_
        self.unseen_subjects_ = []
        return self

    def predict(self, X, subject=None):
        if subject is None:
            raise ValueError("EuclideanAlignedClassifier needs per-trial `subject` ids")
        subject = np.asarray(subject)
        covs = self._covs(X)
        refs = dict(self.refs_)
        for s in np.unique(subject):
            if s not in refs:  # unseen person: unsupervised calibration from their own trials
                refs[s] = covs[subject == s].mean(axis=0)
                self.unseen_subjects_.append(s)
        return self.clf_.predict(self._align(covs, subject, refs))


def riemann_ea_lr(C: float = 1.0, seed: int = 0) -> EuclideanAlignedClassifier:
    return EuclideanAlignedClassifier(C=C, seed=seed)

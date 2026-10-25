"""Honest chance levels.

With a finite test set, a random classifier exceeds 1/k by luck surprisingly often.
Combrisson & Jerbi (2015, J. Neurosci. Methods 250:126-136) give the accuracy a classifier
must beat to be significant at level alpha: the (1 - alpha) quantile of Binomial(n, 1/k), over n.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom


def chance_threshold(n_trials: int, n_classes: int, alpha: float = 0.05) -> float:
    """Smallest accuracy that is significantly above chance (one-sided, level alpha) for n_trials test trials."""
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    return float(binom.ppf(1 - alpha, n_trials, 1.0 / n_classes) / n_trials)


def binomial_pvalue(n_correct: int, n_trials: int, n_classes: int) -> float:
    """P(X >= n_correct) for X ~ Binomial(n_trials, 1/k): the one-sided p-value against guessing."""
    return float(binom.sf(n_correct - 1, n_trials, 1.0 / n_classes))


def permutation_pvalue(observed: float, null_scores) -> float:
    """(count(null >= observed) + 1) / (n_perm + 1); the +1 counts the observed labelling itself, so p is never 0."""
    null_scores = np.asarray(null_scores, dtype=float)
    if null_scores.size == 0:
        raise ValueError("null_scores is empty")
    return float((np.sum(null_scores >= observed) + 1) / (null_scores.size + 1))

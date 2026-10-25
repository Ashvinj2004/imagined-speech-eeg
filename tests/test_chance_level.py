import numpy as np
import pytest

from bci.eval.chance_level import binomial_pvalue, chance_threshold, permutation_pvalue


@pytest.mark.parametrize("k", [2, 3, 5])
def test_threshold_above_one_over_k_for_small_n(k):
    for n in [20, 40, 100]:
        assert chance_threshold(n, k) > 1 / k


def test_threshold_known_value():
    # 2 classes, 40 trials: binom.ppf(0.95, 40, 0.5) = 25 -> 62.5%, well above 50%
    assert chance_threshold(40, 2) == pytest.approx(25 / 40)


@pytest.mark.parametrize("k", [2, 5])
def test_threshold_converges_to_one_over_k(k):
    gaps = [chance_threshold(n, k) - 1 / k for n in [100, 1_000, 10_000, 100_000]]
    assert all(a > b for a, b in zip(gaps, gaps[1:]))
    assert gaps[-1] < 0.005


def test_threshold_shrinks_with_alpha():
    assert chance_threshold(50, 3, alpha=0.01) > chance_threshold(50, 3, alpha=0.05)


@pytest.mark.parametrize("n,k", [(40, 2), (60, 3), (100, 5)])
def test_false_positive_rate_close_to_alpha(n, k):
    """A guessing classifier should cross the threshold at most ~alpha of the time (binomial is discrete, so <=)."""
    rng = np.random.default_rng(0)
    alpha, n_sim = 0.05, 20_000
    accs = rng.binomial(n, 1 / k, size=n_sim) / n
    fpr = np.mean(accs > chance_threshold(n, k, alpha))
    assert 0.01 < fpr <= alpha + 0.005


def test_binomial_pvalue():
    assert binomial_pvalue(0, 50, 2) == pytest.approx(1.0)
    assert binomial_pvalue(50, 50, 2) == pytest.approx(0.5**50)
    # Exceeding the threshold <=> p < alpha
    n, k = 60, 3
    thr_correct = round(chance_threshold(n, k) * n)
    assert binomial_pvalue(thr_correct + 1, n, k) < 0.05
    assert binomial_pvalue(thr_correct, n, k) >= 0.05


def test_permutation_pvalue_correction():
    assert permutation_pvalue(0.9, np.zeros(99)) == pytest.approx(1 / 100)  # never exactly 0
    assert permutation_pvalue(0.1, np.ones(99)) == pytest.approx(1.0)
    assert permutation_pvalue(0.5, [0.4, 0.5, 0.6]) == pytest.approx(3 / 4)  # ties count against us

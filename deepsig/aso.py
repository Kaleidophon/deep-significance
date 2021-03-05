"""
Re-implementation of Almost Stochastic Order (ASO) by [Dror et al. (2019)](https://arxiv.org/pdf/2010.03039.pdf).
The code here heavily borrows from their [original code base](https://github.com/rtmdrr/DeepComparison).
"""

# STD
from typing import List, Callable, Tuple
from warnings import warn

# EXT
import numpy as np
from scipy.stats import norm as normal

# PKG
from deepsig.conversion import ArrayLike, score_conversion


@score_conversion
def aso(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    confidence_level: float = 0.05,
    num_samples_a: int = 1000,
    num_samples_b: int = 1000,
    num_bootstrap_iterations: int = 1000,
    dt: float = 0.005,
) -> Tuple[float, float]:
    """
    Performs the Almost Stochastic Order test by Dror et al. (2019). The function takes two list of scores as input
    (they do not have to be of the same length) and returns the violation ratio and the minimum epsilon
    threshold. If the violation ratio is below the minimum epsilon threshold, the null hypothesis can be rejected
    (and the model scores_a belongs to is deemed better than the model of scores_b). Intuitively, the violation ratio
    denotes the degree to which total stochastic order (algorithm A is *always* better than B) is being violated.

    The epsilon threshold directly depends on the number of supplied scores; thus, the more scores are used, the more
    safely we can reject the null hypothesis.

    Parameters
    ----------
    scores_a: List[float]
        Scores of algorithm A.
    scores_b: List[float]
        Scores of algorithm B.
    confidence_level: float
        Desired confidence level of test. Set to 0.05 by default.
    num_samples_a: int
        Number of samples from the score distribution of A during every bootstrap iteration when estimating sigma.
    num_samples_b: int
        Number of samples from the score distribution of B during every bootstrap iteration when estimating sigma.
    num_bootstrap_iterations: int
        Number of bootstrap iterations when estimating sigma.
    dt: float
        Differential for t during integral calculation.

    Returns
    -------
    Tuple[float, float]
        Return violation ratio and the minimum epsilon threshold. The violation ratio should fall below the threshold in
        order for the null hypothesis to be rejected.
    """
    violation_ratio = compute_violation_ratio(scores_a, scores_b, dt)
    const = np.sqrt(num_samples_a * num_samples_b / (num_samples_a + num_samples_b))
    quantile_func_a = get_quantile_function(scores_a)
    quantile_func_b = get_quantile_function(scores_b)

    samples = np.zeros(num_bootstrap_iterations)
    for i in range(num_bootstrap_iterations):
        sampled_scores_a = quantile_func_a(np.random.uniform(0, 1, num_samples_a))
        sampled_scores_b = quantile_func_b(np.random.uniform(0, 1, num_samples_b))
        samples[i] = compute_violation_ratio(sampled_scores_a, sampled_scores_b, dt)

    sigma_hat = np.std(const * (samples - violation_ratio))

    min_epsilon = min(
        max(
            violation_ratio - (1 / const) * sigma_hat * normal.ppf(confidence_level), 0
        ),
        1,
    )

    return violation_ratio, min_epsilon


def compute_violation_ratio(scores_a: np.array, scores_b: np.array, dt: float) -> float:
    """
    Compute the violation ration e_W2 (equation 4 + 5).

    Parameters
    ----------
    scores_a: List[float]
        Scores of algorithm A.
    scores_b: List[float]
        Scores of algorithm B.
    dt: float
        Differential for t during integral calculation.

    Returns
    -------
    float
        Return violation ratio.
    """
    squared_wasserstein_dist = 0
    int_violation_set = 0  # Integral over violation set A_X
    quantile_func_a = get_quantile_function(scores_a)
    quantile_func_b = get_quantile_function(scores_b)

    for p in np.arange(0, 1, dt):
        diff = quantile_func_b(p) - quantile_func_a(p)
        squared_wasserstein_dist += (diff ** 2) * dt
        int_violation_set += (max(diff, 0) ** 2) * dt

    if squared_wasserstein_dist == 0:
        warn("Division by zero encountered in violation ratio.")
        violation_ratio = 0

    else:
        violation_ratio = int_violation_set / squared_wasserstein_dist

    return violation_ratio


def get_quantile_function(scores: np.array) -> Callable:
    """
    Return the quantile function corresponding to an empirical distribution of scores.

    Parameters
    ----------
    scores: List[float]
        Empirical distribution of scores belonging to an algorithm.

    Returns
    -------
    Callable
        Return the quantile function belonging to an empirical score distribution.
    """

    def _quantile_function(p: float) -> float:
        cdf = np.sort(scores)
        num = len(scores)
        index = int(np.ceil(num * p))

        return cdf[min(num - 1, max(0, index - 1))]

    return np.vectorize(_quantile_function)

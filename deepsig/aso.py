"""
Re-implementation of Almost Stochastic Order (ASO) by [Dror et al. (2019)](https://arxiv.org/pdf/2010.03039.pdf).
The code here heavily borrows from their [original code base](https://github.com/rtmdrr/DeepComparison).
"""

# STD
from typing import List, Callable, Tuple

# EXT
import numpy as np
from scipy.stats import norm as normal


def aso(
    scores_a: List[float],
    scores_b: List[float],
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

    samples = []
    for _ in range(num_bootstrap_iterations):
        sampled_scores_a = list(
            map(get_quantile_function(scores_a), np.random.uniform(0, 1, num_samples_a))
        )
        sampled_scores_b = list(
            map(get_quantile_function(scores_b), np.random.uniform(0, 1, num_samples_b))
        )
        distance = compute_violation_ratio(sampled_scores_a, sampled_scores_b, dt)
        samples.append(distance)

    sigma_hat = np.std(samples)
    min_epsilon = min(
        max(
            violation_ratio - (1 / const) * sigma_hat + normal.ppf(confidence_level), 0
        ),
        1,
    )

    return violation_ratio, min_epsilon


def compute_violation_ratio(
    scores_a: List[float], scores_b: List[float], dt: float
) -> float:
    """
    Compute the violation ration (equation 4 + 5).

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

    for p in np.arange(0, 1, dt):
        diff = get_quantile_function(scores_b)(p) - get_quantile_function(scores_a)(p)
        squared_wasserstein_dist += dt * diff ** 2
        int_violation_set += dt * max(diff, 0) ** 2

    violation_ratio = int_violation_set / (squared_wasserstein_dist + 1e-8)

    return violation_ratio


def get_quantile_function(scores: List[float]) -> Callable:
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

    def _quantile_function(p: int) -> float:
        cdf = np.sort(scores)
        num = len(scores)
        index = int(np.ceil(p * num))

        return cdf[min(num - 1, max(0, index - 1))]

    return _quantile_function

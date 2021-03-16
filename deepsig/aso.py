"""
Re-implementation of Almost Stochastic Order (ASO) by `Dror et al. (2019) <https://arxiv.org/pdf/2010.03039.pdf>`_.
The code here heavily borrows from their `original code base <https://github.com/rtmdrr/DeepComparison>`_.
"""

# STD
from typing import List, Callable, Optional
from warnings import warn

# EXT
import numpy as np
from scipy.stats import norm as normal
from tqdm import tqdm

# PKG
from deepsig.conversion import ArrayLike, score_conversion

# CONST
# Warn the user when min_eps score lands in [0.5 - FAST_QUANTILE_WARN_INTERVAL, 0.5]
# build_quantile="fast" trades of precision for speed but could be misleading for close cases, where
# build_quantile="exact" should be used instead.
FAST_QUANTILE_WARN_INTERVAL = 0.1


@score_conversion
def aso(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    confidence_level: float = 0.05,
    num_samples: int = 1000,
    num_bootstrap_iterations: int = 1000,
    dt: float = 0.005,
    build_quantile: str = "fast",
    show_progress: bool = True,
) -> float:
    """
    Performs the Almost Stochastic Order test by Dror et al. (2019). The function takes two list of scores as input
    (they do not have to be of the same length) and returns an upper bound to the violation ratio - the minimum epsilon
    threshold. If the violation ratio is below 0.5, the null hypothesis can be rejected safely (and the model scores_a
    belongs to is deemed better than the model of scores_b). Intuitively, the violation ratio denotes the degree to
    which total stochastic order (algorithm A is *always* better than B) is being violated.
    The more scores and the higher num_samples / num_bootstrap_iterations, the more reliable is the result.

    Parameters
    ----------
    scores_a: List[float]
        Scores of algorithm A.
    scores_b: List[float]
        Scores of algorithm B.
    confidence_level: float
        Desired confidence level of test. Set to 0.05 by default.
    num_samples: int
        Number of samples from the score distributions during every bootstrap iteration when estimating sigma.
    num_bootstrap_iterations: int
        Number of bootstrap iterations when estimating sigma.
    dt: float
        Differential for t during integral calculation.
    build_quantile: str
        Indicate whether the quantile function during bootstrap iterations should be build from scratch
        (build_quantile="exact") or reused from the original score distribution (build_quantile="fast"). The fast result
        will vary slightly from the exact result, so use "exact" for eps_min close to 0.5.
    show_progress: bool
        Show progress bar. Default is True.

    Returns
    -------
    float
        Return an upper bound to the violation ratio. If it falls below 0.5, the null hypothesis can be rejected.
    """
    assert (
        len(scores_a) > 0 and len(scores_b) > 0
    ), "Both lists of scores must be non-empty."
    assert num_samples > 0, "num_samples must be positive, {} found.".format(
        num_samples
    )
    assert (
        num_bootstrap_iterations > 0
    ), "num_samples must be positive, {} found.".format(num_bootstrap_iterations)
    assert build_quantile in (
        "fast",
        "exact",
    ), "'build_quantile' has to be eiter 'fast' or 'exact', {} found".format(
        build_quantile
    )

    violation_ratio = compute_violation_ratio(scores_a, scores_b, dt)
    const = np.sqrt(num_samples ** 2 / 2 * num_samples)
    quantile_func_a = get_quantile_function(scores_a)
    quantile_func_b = get_quantile_function(scores_b)

    samples = np.zeros(num_bootstrap_iterations)
    iters = (
        tqdm(range(num_bootstrap_iterations), desc="Bootstrap iterations")
        if show_progress
        else range(num_bootstrap_iterations)
    )
    for i in iters:
        sampled_scores_a = quantile_func_a(np.random.uniform(0, 1, num_samples))
        sampled_scores_b = quantile_func_b(np.random.uniform(0, 1, num_samples))
        samples[i] = compute_violation_ratio(
            sampled_scores_a,
            sampled_scores_b,
            dt,
            quantile_func_a if build_quantile == "fast" else None,
            quantile_func_b if build_quantile == "fast" else None,
        )

    sigma_hat = np.std(const * (samples - violation_ratio))

    min_epsilon = min(
        max(
            violation_ratio - (1 / const) * sigma_hat * normal.ppf(confidence_level), 0
        ),
        1,
    )

    if (
        0.5 - FAST_QUANTILE_WARN_INTERVAL < min_epsilon < 0.5
        and build_quantile == "fast"
    ):
        warn(
            "min_epsilon was {:.4f}, but build_quantile='fast' is potentially inaccurate up to {:.2f}. Run the "
            "aso() again with build_quantile='exact' to make sure the null hypothesis can be rejected "
            "safely.".format(min_epsilon, FAST_QUANTILE_WARN_INTERVAL)
        )

    return min_epsilon


def compute_violation_ratio(
    scores_a: np.array,
    scores_b: np.array,
    dt: float,
    quantile_func_a: Optional[Callable] = None,
    quantile_func_b: Optional[Callable] = None,
) -> float:
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
    quantile_func_a: Optional[Callable]
        If quantile_func_a is given, it will not be rebuild using scores_a. Used when build_quantile="fast".
    quantile_func_b: Optional[Callable]
        If quantile_func_b is given, it will not be rebuild using scores_b. Used when build_quantile="fast".

    Returns
    -------
    float
        Return violation ratio.
    """
    squared_wasserstein_dist = 0
    int_violation_set = 0  # Integral over violation set A_X

    if quantile_func_a is None:
        quantile_func_a = get_quantile_function(scores_a)

    if quantile_func_b is None:
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

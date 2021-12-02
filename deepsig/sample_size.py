"""
Implement functions to help determine the right sample size for experiments.
"""

# STD
from math import sqrt
from typing import Callable, Optional

# EXT
import numpy as np
from scipy.stats import ttest_ind
from tqdm import tqdm

# PROJECT
from deepsig.conversion import score_conversion, ArrayLike


def aso_uncertainty_reduction(m_old: int, n_old: int, m_new: int, n_new: int) -> float:
    """
    Compute the reduction of uncertainty of tightness of estimate for violation ratio e_W2(F, G).
    This is based on the CLT in `del Barrio et al. (2018) <https://arxiv.org/pdf/1705.01788.pdf>`_ Theorem 2.4 / eq. 9.

    Parameters
    ----------
    m_old: int
        Old number of scores for algorithm A.
    n_old: int
        Old number of scores for algorithm B.
    m_new: int
        New number of scores for algorithm A.
    n_new: int
        New number of scores for algorithm B.

    Returns
    -------
    float
        Reduction of uncertainty / increase of tightness of estimate for violation ratio e_W2(F, G).

    """
    assert all(
        sample_size >= 1 for sample_size in [m_old, n_old, m_new, n_new]
    ), "All sample sizes have to be larger than or equal to 1."

    assert all(
        type(sample_size) == int for sample_size in [m_old, n_old, m_new, n_new]
    ), "Sample sizes have to be integers."

    return sqrt((m_old + n_old) * m_new * n_new / (m_old * n_old * (m_new + n_new)))


@score_conversion
def bootstrap_power_analysis(
    scores: ArrayLike,
    scalar: float = 1.25,
    num_bootstrap_iterations: int = 5000,
    significance_threshold: float = 0.05,
    significance_test: Optional[Callable] = None,
    show_progress: bool = True,
    seed: Optional[int] = None,
) -> float:
    """
    Perform bootstrap power analysis [1] to see whether the amount of collected scores is sufficient. It determines
    the statistical power of the sample, i.e. the probability of an statistically significant effect to be found given
    that there is one (that is, the lower the power, the higher the probability of a Type II error).

    This is done by giving all scores a uniform lift as suggested by [2] by using the value in the `scalar` named
    argument. Then, a number of bootstrap iterations is run and a significance test to see whether the difference
    between the original and the scaled scores is  significant. If the percentage of significant comparisons is low,
    more samples should be gathered - a uniform lift of 1.25 should result in many significant differences, thus the
    variance in the original sample is too high.

    By default, a one-tailed Welch's t-test is used. However, this can be changed by supplying a different
    `significance_test` named argument (the test might have to be wrapped in a lambda function or similar to only
    return the p-value, not the test statistic).

    [1] Yuan, Ke‐Hai, and Kentaro Hayashi. "Bootstrap approach to inference and power analysis based on three test
    statistics for covariance structure models." British Journal of Mathematical and Statistical Psychology 56.1 (2003):
    93-110.
    [2] Henderson, Peter, et al. "Deep reinforcement learning that matters." Proceedings of the AAAI conference on
    artificial intelligence. Vol. 32. No. 1. 2018.

    Parameters
    ----------
    scores: ArrayLike
        Scores to be examined.
    scalar: float
        Scalar used for lifting scores.
    num_bootstrap_iterations: int
        Number of bootstrap iterations.
    significance_threshold: float
        Significance threshold to determine whether a comparison is significant.
    significance_test: Optional[Callable]
        Callable function returning a p-value (or similar) based on two sets of scores. If None, a Welch's t-test is
        used.
    show_progress: bool
        Show progress bar. Default is True.
    seed: Optional[int]
        Set seed for reproducibility purposes. Default is None (meaning no seed is used).

    Returns
    -------
    float:
        Percentage of significant comparisons. If the percentage is low, more samples should be gathered.
    """
    assert len(scores) > 0, "Lists of scores must be non-empty."
    assert (
        num_bootstrap_iterations > 0
    ), "Number of bootstrap iterations should be positive"
    assert (
        scalar > 1
    ), "Lift should be larger than 1 to produce significant differences."

    if seed is not None:
        np.random.seed(seed)

    # Set default significance test to Welch's t-test
    if significance_test is None:
        significance_test = lambda scores_a, scores_b: ttest_ind(
            scores_a, scores_b, equal_var=False, alternative="greater"
        ).pvalue

    iters = (
        range(num_bootstrap_iterations)
        if not show_progress
        else tqdm(range(num_bootstrap_iterations))
    )

    # Instead of just multiplying, do this so that negative values are lifted as well and not amplified
    scores_lifted = scores + abs(scores) * (scalar - 1)
    N = len(scores)
    num_significant = 0

    for _ in iters:
        resampled_scores = np.random.choice(scores, N)
        resampled_scores_lifted = np.random.choice(scores_lifted, N)

        p_value = significance_test(resampled_scores_lifted, resampled_scores)
        num_significant += int(p_value <= significance_threshold)

    num_significant /= num_bootstrap_iterations

    return num_significant

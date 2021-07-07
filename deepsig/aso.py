"""
Re-implementation of Almost Stochastic Order (ASO) by `Dror et al. (2019) <https://arxiv.org/pdf/2010.03039.pdf>`_.
The code here heavily borrows from their `original code base <https://github.com/rtmdrr/DeepComparison>`_.

References:
-----------
[1] Dror, Rotem, Segev Shlomov, and Roi Reichart. "Deep dominance-how to properly compare deep neural models."
Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019.
[2] Hoffman, Matthew D., and Andrew Gelman. "The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian
Monte Carlo." J. Mach. Learn. Res. 15.1 (2014): 1593-1623.
"""

# STD
from typing import List, Callable, Tuple, Type, Dict
from warnings import warn

# EXT
from joblib import Parallel, delayed
import numpy as np
import pymc3.distributions.continuous as dists
from pymc3.distributions.distribution import Distribution
from pymc3.sampling import NUTS
from pymc3.step_methods.hmc.base_hmc import BaseHMC
from scipy.stats import norm as normal
from tqdm import tqdm

# PKG
from deepsig.conversion import ArrayLike, score_conversion


@score_conversion
def aso(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    confidence_level: float = 0.05,
    num_samples: int = 1000,
    num_bootstrap_iterations: int = 1000,
    dt: float = 0.005,
    num_jobs: int = 1,
    show_progress: bool = True,
) -> float:
    """
    Performs the Almost Stochastic Order test by Dror et al. (2019). The function takes two list of scores as input
    (they do not have to be of the same length) and returns an upper bound to the violation ratio - the minimum epsilon
    threshold. `scores_a` should contain scores of the algorithm which we suspect to be better (in this setup,
    higher = better).

    The null hypothesis (which we would like to reject), is that the algorithm that generated `scores_a` is
    *not* better than the one `scores_b` originated from. If the violation ratio is below 0.5, the null hypothesis can
    be rejected safely (and the model scores_a belongs to is deemed better than the model of scores_b). Intuitively, the
    violation ratio denotes the degree to which total stochastic order (algorithm A is *always* better than B) is being
    violated. The more scores and the higher num_samples / num_bootstrap_iterations, the more reliable is the result.

    Parameters
    ----------
    scores_a: ArrayLike
        Scores of algorithm A.
    scores_b: ArrayLike
        Scores of algorithm B.
    confidence_level: float
        Desired confidence level of test. Set to 0.05 by default.
    num_samples: int
        Number of samples from the score distributions during every bootstrap iteration when estimating sigma.
    num_bootstrap_iterations: int
        Number of bootstrap iterations when estimating sigma.
    dt: float
        Differential for t during integral calculation.
    num_jobs: int
        Number of threads that bootstrap iterations are divided among.
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
    assert num_jobs > 0, "Number of jobs has to be at least 1, {} found.".format(
        num_jobs
    )

    const1 = np.sqrt(len(scores_a) * len(scores_b) / (len(scores_a) + len(scores_b)))

    violation_ratio, sigma_hat = get_bootstrap_estimates(
        scores_a,
        scores_b,
        num_samples,
        num_bootstrap_iterations,
        dt,
        num_jobs,
        show_progress,
    )

    # Compute eps_min and make sure it stays in [0, 1]
    min_epsilon = min(
        max(
            violation_ratio - (1 / const1) * sigma_hat * normal.ppf(confidence_level), 0
        ),
        1,
    )

    return min_epsilon


@score_conversion
def bf_aso(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    prior: Type = dists.Beta,
    prior_args: Dict[str, float] = {"alpha": 1, "beta": 1},
    num_samples: int = 1000,
    num_bootstrap_iterations: int = 1000,
    dt: float = 0.005,
    sampler: BaseHMC = NUTS,
    num_jobs: int = 1,
    show_progress: bool = True,
) -> float:
    """
    Compute the Bayes factor BF_01 for Almost stochastic order, where the null hypothesis H_0: e_W2 = 0.5 and the
    alternate hypothesis H_1: e_W2 =/= 0.5.

    Parameters
    ----------
    scores_a: ArrayLike
        Scores of algorithm A.
    scores_b: ArrayLike
        Scores of algorithm B.
    prior: Type
        Prior distribution for the violation ratio. Set to a Beta(1, 1) by default (so a uniform prior; see prior_args).
    prior_args: Dict[str, float]
        Dictionary of arguments to instantiate prior.
    num_samples: int
        Number of samples from the score distributions during every bootstrap iteration when estimating sigma.
    num_bootstrap_iterations: int
        Number of bootstrap iterations when estimating sigma.
    dt: float
        Differential for t during integral calculation.
    sampler: BaseHMC
        MCMC sampler used. Defaults to the No-U-Turn-Sampler (NUTS) by [2].
    num_jobs: int
        Number of threads that bootstrap iterations are divided among. Also, number of chains used for the MCMC sampler.
    show_progress: bool
        Show progress bar. Default is True.

    Returns
    -------
    float
        Bayes factor BF_01.
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
    assert num_jobs > 0, "Number of jobs has to be at least 1, {} found.".format(
        num_jobs
    )
    ...  # TODO: Implement


def get_bootstrap_estimates(
    scores_a: np.array,
    scores_b: np.array,
    num_samples: int,
    num_bootstrap_iterations: int,
    dt: float,
    num_jobs: int,
    show_progress: bool,
):
    """
    Get the bootstrap estimates of the violation ratio and the associated variance sigma_hat.

    Parameters
    ----------
    scores_a: ArrayLike
        Scores of algorithm A.
    scores_b: ArrayLike
        Scores of algorithm B.
    num_samples: int
        Number of samples from the score distributions during every bootstrap iteration when estimating sigma.
    num_bootstrap_iterations: int
        Number of bootstrap iterations when estimating sigma.
    dt: float
        Differential for t during integral calculation.
    num_jobs: int
        Number of threads that bootstrap iterations are divided among.
    show_progress: bool
        Show progress bar. Default is True.

    Returns
    -------
    Tuple[float, float]
        Estimated violation ratio and associated variance.
    """
    violation_ratio = compute_violation_ratio(scores_a, scores_b, dt)
    # Based on the actual number of samples
    quantile_func_a = get_quantile_function(scores_a)
    quantile_func_b = get_quantile_function(scores_b)

    # Add progressbar if applicable
    iters = (
        tqdm(range(num_bootstrap_iterations), desc="Bootstrap iterations")
        if show_progress
        else range(num_bootstrap_iterations)
    )

    def _bootstrap_iter():
        """
        One bootstrap iteration. Wrapped in a function so it can be handed to joblib.Parallel.
        """
        sampled_scores_a = quantile_func_a(np.random.uniform(0, 1, num_samples))
        sampled_scores_b = quantile_func_b(np.random.uniform(0, 1, num_samples))
        sample = compute_violation_ratio(
            sampled_scores_a,
            sampled_scores_b,
            dt,
        )

        return sample

    # Initialize worker pool and start iterations
    parallel = Parallel(n_jobs=num_jobs)
    samples = parallel(delayed(_bootstrap_iter)() for _ in iters)

    const2 = np.sqrt(
        num_samples ** 2 / (2 * num_samples)
    )  # This one is based on the number of re-sampled scores
    sigma_hat = np.std(const2 * (samples - violation_ratio))

    return violation_ratio, sigma_hat


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

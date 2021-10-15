"""
Re-implementation of Almost Stochastic Order (ASO) by `Dror et al. (2019) <https://arxiv.org/pdf/2010.03039.pdf>`_.
The code here heavily borrows from their `original code base <https://github.com/rtmdrr/DeepComparison>`_.
"""

# STD

from typing import List, Callable, Union, Optional
from warnings import warn

# EXT
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
import numpy as np
import pandas as pd
from scipy.stats import norm as normal
from tqdm import tqdm

# PKG
from deepsig.conversion import (
    ArrayLike,
    ScoreCollection,
    score_conversion,
    ALLOWED_TYPES,
    CONVERSIONS,
)

# MISC
set_loky_pickler("dill")  # Avoid weird joblib error with multi_aso


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
    seed: Optional[int] = None,
    _progress_bar: Optional[tqdm] = None,
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
    num_jobs: int
        Number of threads that bootstrap iterations are divided among.
    show_progress: bool
        Show progress bar. Default is True.
    seed: Optional[int]
        Set seed for reproducibility purposes. Default is None (meaning no seed is used).
    _progress_bar: Optional[tqdm]
        Hands over a progress bar object when called by multi_aso(). Only for internal use.

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

    violation_ratio = compute_violation_ratio(scores_a, scores_b, dt)
    # Based on the actual number of samples
    const1 = np.sqrt(len(scores_a) * len(scores_b) / (len(scores_a) + len(scores_b)))
    quantile_func_a = get_quantile_function(scores_a)
    quantile_func_b = get_quantile_function(scores_b)

    def _progress_iter(high: int, progress_bar: tqdm):
        """
        This function is used when a shared progress bar is passed from multi_aso() - every time the iterator yields an
        element, the progress bar is updated by one. It essentially behaves like a simplified range() function.

        Parameters
        ----------
        high: int
            Number of elements in iterator.
        progress_bar: tqdm
            Shared progress bar.
        """
        current = 0

        while current < high:
            yield current
            current += 1
            progress_bar.update(1)

    # Add progress bar if applicable
    if show_progress and _progress_bar is None:
        iters = tqdm(range(num_bootstrap_iterations), desc="Bootstrap iterations")

    # Shared progress bar when called from multi_aso()
    elif _progress_bar is not None:
        iters = _progress_iter(num_bootstrap_iterations, _progress_bar)

    else:
        iters = range(num_bootstrap_iterations)

    # Set seeds for different jobs if applicable
    # "Sub-seeds" for jobs are just seed argument + job index
    seeds = (
        [None] * num_jobs
        if seed is None
        else [seed + offset for offset in range(1, num_jobs + 1)]
    )

    def _bootstrap_iter(seed: Optional[int] = None):
        """
        One bootstrap iteration. Wrapped in a function so it can be handed to joblib.Parallel.
        """
        if seed is not None:
            np.random.seed(seed)

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
    samples = parallel(delayed(_bootstrap_iter)(seed) for seed, _ in zip(seeds, iters))

    const2 = np.sqrt(
        num_samples ** 2 / (2 * num_samples)
    )  # This one is based on the number of re-sampled scores
    sigma_hat = np.std(const2 * (samples - violation_ratio))

    # Compute eps_min and make sure it stays in [0, 1]
    min_epsilon = min(
        max(
            violation_ratio - (1 / const1) * sigma_hat * normal.ppf(confidence_level), 0
        ),
        1,
    )

    return min_epsilon


def multi_aso(
    scores: ScoreCollection,
    confidence_level: float = 0.05,
    use_bonferroni: bool = True,
    use_symmetry: bool = True,
    num_samples: int = 1000,
    num_bootstrap_iterations: int = 1000,
    dt: float = 0.005,
    num_jobs: int = 1,
    return_df: bool = False,
    show_progress: bool = True,
    seed: Optional[int] = None,
) -> Union[np.array, pd.DataFrame]:
    """
    Provides easy function to compare the scores of multiple models at ones. Scores can be supplied in various forms
    (dictionary, nested list, 2D arrays or tensors). Returns a matrix (or pandas.DataFrame) with results. Applies
    Bonferroni correction to confidence level by default, but can be disabled by use_bonferroni=False.

    Parameters
    ----------
    scores: ScoreCollection
        Collection of model scores. Should be either dictionary of model name to model scores, nested Python list,
        2D numpy or Jax array, or 2D Tensorflow or PyTorch tensor.
    confidence_level: float
        Desired confidence level of test. Set to 0.05 by default.
    use_bonferroni: bool
        Indicate whether Bonferroni correction should be applied to confidence level in order to adjust for the number
        of comparisons. Default is True.
    use_symmetry: bool
        Use the fact that ASO(A, B, alpha) = 1 - ASO(B, A, alpha)
        `del Barrio et al. (2018) <https://arxiv.org/pdf/1705.01788.pdf>`_ to save half of the computations. Default is
        True.
    num_samples: int
        Number of samples from the score distributions during every bootstrap iteration when estimating sigma.
    num_bootstrap_iterations: int
        Number of bootstrap iterations when estimating sigma.
    dt: float
        Differential for t during integral calculation.
    num_jobs: int
        Number of threads that bootstrap iterations are divided among.
    return_df: bool
        Indicate whether result should be returned as pandas DataFrame. Only possible if scores is a dictionary of
        model names to model scores. Otherwise, 2D numpy array with eps_min scores is returned. Default is False.
    show_progress: bool
        Show progress bar. Default is True.
    seed: Optional[int]
        Set seed for reproducibility purposes. Default is None (meaning no seed is used).

    Returns
    -------
    Union[np.array, pd.DataFrame]
        2D numpy array or pandas Dataframe (if scores is dictionary and return_df=True) with result of ASO.
    """
    num_models = _get_num_models(scores)
    num_comparisons = num_models * (num_models - 1) / 2
    eps_min = np.eye(num_models)  # Initialize score matrix

    if use_bonferroni:
        confidence_level /= num_comparisons

    # Iterate over simple indices or dictionary keys depending on type of scores argument
    indices = list(range(num_models)) if type(scores) != dict else list(scores.keys())

    # Add progressbar if applicable
    progress_bar = None
    if show_progress:
        progress_bar = tqdm(
            range(int(num_comparisons * num_bootstrap_iterations))
            if use_symmetry
            else range(int(num_comparisons * num_bootstrap_iterations * 2)),
            desc="Model comparisons",
        )

    for i, key_i in enumerate(indices):
        for j, key_j in enumerate(indices[(i + 1) :]):
            scores_a, scores_b = scores[key_i], scores[key_j]

            eps_min[i, j] = aso(
                scores_a,
                scores_b,
                confidence_level=confidence_level,
                num_samples=num_samples,
                num_bootstrap_iterations=num_bootstrap_iterations,
                dt=dt,
                num_jobs=num_jobs,
                show_progress=False,
                seed=seed,
                _progress_bar=progress_bar,
            )

            # Use ASO(A, B, alpha) = 1 - ASO(B, A, alpha)
            if use_symmetry:
                eps_min[j, i] = eps_min[i, j]

            # Compute ASO(B, A, alpha) separately
            else:
                eps_min[i, j] = aso(
                    scores_b,
                    scores_a,
                    confidence_level=confidence_level,
                    num_samples=num_samples,
                    num_bootstrap_iterations=num_bootstrap_iterations,
                    dt=dt,
                    num_jobs=num_jobs,
                    show_progress=False,
                    seed=seed,
                    _progress_bar=progress_bar,
                )

    if type(scores) == dict and return_df:
        eps_min = pd.DataFrame(data=eps_min, index=list(scores.keys()))
        eps_min = eps_min.rename(dict(enumerate(scores.keys())), axis=1)

    return eps_min


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


def _get_num_models(scores: ScoreCollection) -> int:
    """
    Retrieve the number of models from a ScoreCollection for multi_aso().

    Parameters
    ----------
    scores: ScoreCollection
        Collection of model scores. Should be either dictionary of model name to model scores, nested Python list,
        2D numpy or Jax array, or 2D Tensorflow or PyTorch tensor.

    Returns
    -------
    int
        Number of models.
    """
    # Python dictionary
    if isinstance(scores, dict):
        if len(scores) < 2:
            raise ValueError(
                "'scores' argument should contain at least two sets of scores, but only {} found.".format(
                    len(scores)
                )
            )

        return len(scores)

    # (Nested) python list
    elif isinstance(scores, list):
        if not isinstance(scores[0], list):
            raise TypeError(
                "'scores' argument must be nested list of scores when Python lists are used, but elements of type {} "
                "found".format(type(scores[0]).__name__)
            )

        return len(scores)

    # Numpy / Jax arrays, Tensorflow / PyTorch tensor
    elif type(scores) in ALLOWED_TYPES:
        scores = CONVERSIONS[type(scores)](scores)  # Convert to numpy array

        return scores.shape[0]

    raise TypeError(
        "Invalid type for 'scores', should be nested Python list, dict, Jax / Numpy array or Tensorflow / PyTorch "
        "tensor, '{}' found.".format(type(scores).__name__)
    )

# STD
from collections import defaultdict
import math
import os
import scipy
from typing import Optional, Dict, Callable, Any, List
from warnings import warn

# EXT
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm as normal
from tqdm import tqdm
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from deepsig.conversion import score_pair_conversion
from deepsig.aso import ArrayLike, get_quantile_function

# CONST
SAMPLE_SIZE = 20
SAVE_DIR = "./img"
NUM_SIMULATIONS = 500
VARIANT_COLORS = {
    "Classic Bootstrap": "darkred",
    "Dror et al. (2019)": "darkblue",
    r"Bootstrap $\varepsilon_{\mathcal{W}_2}$ mean": "forestgreen",
    "Bootstrap correction": "darkorange",
    "Cond. Bootstrap corr.": "darkviolet",
    "Cond. Bootstrap corr. 2": "slategray",
    "ASO": "darkred",
}

# MISC
set_loky_pickler("dill")  # Avoid weird joblib error with multi_aso


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

    Returns
    -------
    float
        Return violation ratio.
    """
    if quantile_func_a is None:
        quantile_func_a = get_quantile_function(scores_a)

    if quantile_func_b is None:
        quantile_func_b = get_quantile_function(scores_b)

    squared_wasserstein_dist = 0
    int_violation_set = 0  # Integral over violation set A_X

    for p in np.arange(0 + dt, 1 - dt, dt):
        diff = quantile_func_b(p) - quantile_func_a(p)
        squared_wasserstein_dist += (diff ** 2) * dt
        int_violation_set += (max(diff, 0) ** 2) * dt

    if squared_wasserstein_dist == 0:
        warn("Division by zero encountered in violation ratio.")
        violation_ratio = 0.5

    else:
        violation_ratio = int_violation_set / squared_wasserstein_dist

    return violation_ratio


@score_pair_conversion
def aso_bootstrap_comparisons(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    confidence_level: float = 0.05,
    num_samples: int = 1000,
    num_bootstrap_iterations: int = 1000,
    dt: float = 0.005,
    num_jobs: int = 2,
    show_progress: bool = False,
    seed: Optional[int] = None,
    _progress_bar: Optional[tqdm] = None,
) -> Dict[str, float]:
    """
    Like the package ASO function, but compares different choices of bootstrap estimator.

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
        [None] * num_bootstrap_iterations
        if seed is None
        else [
            seed + offset
            for offset in range(1, math.ceil((num_bootstrap_iterations + 1)))
        ]
    )

    def _bootstrap_iter(seed: Optional[int] = None):
        """
        One bootstrap iteration. Wrapped in a function so it can be handed to joblib.Parallel.
        """
        # When running multiple jobs, these modules have to be re-imported for some reason to avoid an error
        # Use dir() to check whether module is available in local scope:
        # https://stackoverflow.com/questions/30483246/how-to-check-if-a-module-has-been-imported
        if "np" not in dir() or "deepsig" not in dir():
            import numpy as np
            from deepsig.aso import compute_violation_ratio

        if seed is not None:
            np.random.seed(seed)

        sampled_scores_a = quantile_func_a(np.random.uniform(0, 1, len(scores_a)))
        sampled_scores_b = quantile_func_b(np.random.uniform(0, 1, len(scores_b)))
        sample = compute_violation_ratio(
            sampled_scores_a,
            sampled_scores_b,
            dt,
        )

        return sample

    # Initialize worker pool and start iterations
    parallel = Parallel(n_jobs=num_jobs)
    samples = parallel(delayed(_bootstrap_iter)(seed) for seed, _ in zip(seeds, iters))

    # Compute the different variants of the bootstrap estimator

    # 1. Classic bootstrap estimator
    sigma_hat1 = np.std(
        1 / (num_bootstrap_iterations - 1) * (samples - np.mean(samples))
    )
    min_epsilon1 = np.clip(
        violation_ratio - (1 / const1) * sigma_hat1 * normal.ppf(confidence_level),
        0,
        1,
    )

    # 2. ASO as implemented by Dror et al. (2019)
    sigma_hat2 = np.std(const1 * (samples - violation_ratio))
    min_epsilon2 = np.clip(
        violation_ratio - (1 / const1) * sigma_hat2 * normal.ppf(confidence_level),
        0,
        1,
    )

    # 3. Like 2., but using the expected violation ratio for sigma
    sigma_hat3 = np.std(const1 * (samples - np.mean(samples)))
    min_epsilon3 = np.clip(
        violation_ratio - (1 / const1) * sigma_hat3 * normal.ppf(confidence_level),
        0,
        1,
    )

    # 4. Like 3, but with the classic bootstrap bias correction
    corrected_bootstrap_violation_ratio = np.clip(
        2 * violation_ratio - np.mean(samples), 0, 1
    )
    min_epsilon4 = np.clip(
        corrected_bootstrap_violation_ratio
        - (1 / const1) * sigma_hat3 * normal.ppf(confidence_level),
        0,
        1,
    )

    # 5. Like 4., but with conditionally corrected bootstrap estimate
    bias = np.mean(samples) - violation_ratio
    sigma_hat_corr = np.std(1 / (len(samples) - 1) * (samples - np.mean(samples)))
    min_epsilon5 = np.clip(
        (
            corrected_bootstrap_violation_ratio
            if bias >= sigma_hat_corr
            else violation_ratio
        )
        - (1 / const1) * sigma_hat3 * normal.ppf(confidence_level),
        0,
        1,
    )

    # 6. Like 5, but conditional correction happens based on the later used sigma hat
    min_epsilon6 = np.clip(
        (corrected_bootstrap_violation_ratio if bias >= sigma_hat3 else violation_ratio)
        - (1 / const1) * sigma_hat3 * normal.ppf(confidence_level),
        0,
        1,
    )

    return {
        "Classic Bootstrap": min_epsilon1,
        "Dror et al. (2019)": min_epsilon2,
        r"Bootstrap $\varepsilon_{\mathcal{W}_2}$ mean": min_epsilon3,
        "Bootstrap correction": min_epsilon4,
        "Cond. Bootstrap corr.": min_epsilon5,
        "Cond. Bootstrap corr. 2": min_epsilon6,
    }


def test_type1_error(
    sample_size: int,
    colors: Dict[str, str],
    name: str,
    num_simulations: int = 200,
    dist_func: Callable = np.random.normal,
    dist_params: Dict[str, Any] = {"loc": 0, "scale": 1.5},
    save_dir: Optional[str] = None,
):
    """
    Test the rate of type I error (false positive) under different samples sizes.

    Parameters
    ----------
    sample_size: int
        Sample size used in experiments.
    colors: Dict[str, str]
        Colors corresponding to each test for plotting.
    name: str
        Name of the experiment.
    num_simulations: int
        Number of simulations conducted.
    dist_func: Callable
        Distribution function that is used for sampling.
    dist_params: Dict[str, Any]
        Parameters of the distribution function.
    save_dir: Optional[str]
        Directory that plots should be saved to.
    """
    simulation_results = defaultdict(list)

    with tqdm(total=len(colors) * num_simulations) as progress_bar:
        for _ in range(num_simulations):

            # Sample scores for this round
            scores_a = dist_func(**dist_params, size=sample_size)
            scores_b = dist_func(**dist_params, size=sample_size)

            results = aso_bootstrap_comparisons(scores_a, scores_b)

            for variant, res in results.items():
                simulation_results[variant].append(res)

            progress_bar.update(len(colors))

    # with open(f"{save_dir}/type1_pg_rates.pkl", "wb") as out_file:
    #    pickle.dump(simulation_results, out_file)

    # Plot Type I error rates as line plot
    plt.figure(figsize=(8, 6))
    plt.rcParams.update(
        {"font.size": 18, "text.usetex": True, "legend.loc": "upper right"}
    )

    # Create datastructure for boxplots
    data = [simulation_results[test_name] for test_name in simulation_results.keys()]

    box_plot = plt.boxplot(
        data,
        widths=0.45,
        patch_artist=True,
    )

    for variant_name, patch, color in zip(
        simulation_results.keys(), box_plot["boxes"], colors.values()
    ):
        patch.set_edgecolor(color)
        patch.set_facecolor("white")

        plt.plot([], color=color, label=variant_name)

    ax = plt.gca()
    ax.set_ylim(0, 1)
    # ax.set_xlim(-2, 3)
    ax.yaxis.grid()
    plt.xlabel("Bootstrap variants")
    plt.ylabel(r"$\varepsilon_\mathrm{min}$")
    plt.legend()

    if save_dir is not None:
        plt.tight_layout()
        plt.savefig(f"{save_dir}/type1_bootstrap_dists_{name}.png")
    else:
        plt.show()

    plt.close()


def test_type2_error(
    sample_size: int,
    colors: Dict[str, str],
    name: str,
    num_simulations: int = 200,
    dist_func: Callable = np.random.normal,
    inv_cdf_func: Callable = scipy.stats.norm.ppf,
    dist_params: Dict[str, Any] = {"loc": 0, "scale": 0.5},
    dist_params2: Dict[str, Any] = {"loc": -0.25, "scale": 1.5},
    save_dir: Optional[str] = None,
):
    """
    Test the rate of type I error (false positive) under different samples sizes.

    Parameters
    ----------
    sample_size: int
        Sample size used in experiments.
    colors: Dict[str, str]
        Colors corresponding to each test for plotting.
    name: str
        Name of the experiment.
    num_simulations: int
        Number of simulations conducted.
    dist_func: Callable
        Distribution function that is used for sampling.
    inv_cdf_funcL Callable
        Inverse cumulative distribution function in order to compute the exact violation ratio.
    dist_params: Dict[str, Any]
        Parameters of the distribution function.
    dist_params2: Dict[str, Any]
        Parameters of the comparison distribution function.
    save_dir: Optional[str]
        Directory that plots should be saved to.
    """
    simulation_results = defaultdict(list)

    with tqdm(total=len(colors) * num_simulations) as progress_bar:
        for _ in range(num_simulations):

            # Sample scores for this round
            scores_a = dist_func(**dist_params, size=sample_size)
            scores_b = dist_func(**dist_params2, size=sample_size)

            results = aso_bootstrap_comparisons(scores_a, scores_b)

            for variant, res in results.items():
                simulation_results[variant].append(res)

            progress_bar.update(len(colors))

    # with open(f"{save_dir}/type1_pg_rates.pkl", "wb") as out_file:
    #    pickle.dump(simulation_results, out_file)

    # Plot Type I error rates as line plot
    plt.figure(figsize=(8, 6))
    plt.rcParams.update(
        {"font.size": 18, "text.usetex": True, "legend.loc": "upper right"}
    )

    # Create datastructure for boxplots
    data = [simulation_results[test_name] for test_name in simulation_results.keys()]

    box_plot = plt.boxplot(
        data,
        widths=0.45,
        patch_artist=True,
    )

    for variant_name, patch, color in zip(
        simulation_results.keys(), box_plot["boxes"], colors.values()
    ):
        patch.set_edgecolor(color)
        patch.set_facecolor("white")

        plt.plot([], color=color, label=variant_name)

    real_violation_ratio = compute_violation_ratio(
        [],
        [],
        dt=0.05,
        quantile_func_a=lambda p: inv_cdf_func(p, **dist_params),
        quantile_func_b=lambda p: inv_cdf_func(p, **dist_params2),
    )

    ax = plt.gca()
    ax.set_ylim(0, 1)
    x = np.arange(ax.get_xlim()[0], ax.get_xlim()[1] + 1)
    plt.plot(
        x,
        np.ones(len(x)) * real_violation_ratio,
        alpha=0.8,
        linestyle="--",
        color="black",
    )
    ax.yaxis.grid()
    plt.xlabel("Bootstrap variants")
    plt.ylabel(r"$\varepsilon_\mathrm{min}$")
    plt.legend()

    if save_dir is not None:
        plt.tight_layout()
        plt.savefig(f"{save_dir}/type2_bootstrap_dists_{name}.png")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    test_type1_error(
        sample_size=SAMPLE_SIZE,
        colors=VARIANT_COLORS,
        num_simulations=NUM_SIMULATIONS,
        name="normal",
        save_dir=SAVE_DIR,
    )

    test_type2_error(
        sample_size=SAMPLE_SIZE,
        colors=VARIANT_COLORS,
        num_simulations=NUM_SIMULATIONS,
        name="normal",
        save_dir=SAVE_DIR,
    )

# STD
import os
from typing import Optional, Dict, Callable, Any, Tuple, List
import pickle

# EXT
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm as normal
from tqdm import tqdm
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from deepsig.conversion import score_pair_conversion
from deepsig.aso import ArrayLike, compute_violation_ratio, get_quantile_function

# CONST
SAMPLE_SIZES = [5, 10, 15, 20]
SAVE_DIR = "./img"
NUM_SIMULATIONS = 100

# MISC
set_loky_pickler("dill")  # Avoid weird joblib error with multi_aso


@score_pair_conversion
def aso_debug(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    confidence_level: float = 0.05,
    num_samples: int = 1000,
    num_bootstrap_iterations: int = 1000,
    dt: float = 0.005,
    num_jobs: int = 4,
    show_progress: bool = False,
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

    # Experimental: New estimator for violation ratio
    psi_func = lambda gamma: quantile_func_a(gamma) - quantile_func_b(gamma)
    mean_term = np.mean(scores_a) - np.mean(scores_b)

    gammas = np.cumsum(psi_func(np.arange(0, 1, dt))) - mean_term

    max_gammas_indices = np.arange(int(1 / dt))[gammas == np.max(gammas)]
    min_gamma = min(gammas[max_gammas_indices])

    violation_ratio_gamma = (
        min_gamma if psi_func(min_gamma) - mean_term >= 0 else 1 - min_gamma
    )

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
        else [seed + offset for offset in range(1, num_bootstrap_iterations + 1)]
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

        sampled_scores_a = quantile_func_a(np.random.uniform(0, 1, num_samples))
        sampled_scores_b = quantile_func_b(np.random.uniform(0, 1, num_samples))

        # # TODOL Use estimator as an argument here
        sample = compute_violation_ratio(
            sampled_scores_a,
            sampled_scores_b,
            "pi",
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
    sigma_hat2 = np.var(1 / const1 * (samples - violation_ratio))  # TODO: Debug
    sigma_hat3 = np.var(1 / const1 * (samples - violation_ratio_gamma))  # TODO: Debug

    # Compute eps_min and make sure it stays in [0, 1]
    min_epsilon = np.clip(
        violation_ratio - (1 / const1) * sigma_hat * normal.ppf(confidence_level), 0, 1
    )
    min_epsilon2 = np.clip(
        violation_ratio - (1 / const1) * sigma_hat2 * normal.ppf(confidence_level), 0, 1
    )  # TODO: Debug
    min_epsilon3 = np.clip(
        violation_ratio_gamma
        - (1 / const1) * sigma_hat3 * normal.ppf(confidence_level),
        0,
        1,
    )  # TODO: Debug

    return min_epsilon, min_epsilon2, min_epsilon3


def test_type1_error(
    sample_sizes: List[int],
    num_simulations: int = 200,
    dist_func: Callable = np.random.normal,
    dist_params: Dict[str, Any] = {"loc": 0, "scale": 1.5},
    threshold: float = 0.05,
    colors_and_markers: Optional[Dict[str, Tuple[str, str]]] = None,
    save_dir: Optional[str] = None,
):
    """
    Test the rate of type I error (false positive) under different samples sizes.

    Parameters
    ----------
    tests: Dict[str, Callable]
        Considered tests.
    sample_sizes: List[int]
        Samples sizes that are being tested.
    num_simulations: int
        Number of simulations conducted.
    dist_func: Callable
        Distribution function that is used for sampling.
    dist_params: Dict[str, Any]
        Parameters of the distribution function.
    threshold: float
        Threshold that test results has to fall below in order for significance to be claimed.
    colors_and_markers: Optional[Dict[str, Tuple[str, str]]]
        Colors and markers corresponding to each test for plotting.
    save_dir: Optional[str]
        Directory that plots should be saved to.
    """
    simulation_results = {
        test_name: {sample_size: [] for sample_size in sample_sizes}
        for test_name in range(3)
    }

    with tqdm(total=len(sample_sizes) * num_simulations) as progress_bar:
        for sample_size in sample_sizes:
            for _ in range(num_simulations):

                # Sample scores for this round
                scores_a = dist_func(**dist_params, size=sample_size)
                scores_b = dist_func(**dist_params, size=sample_size)

                results = aso_debug(scores_a, scores_b)

                for i, res in enumerate(results):
                    simulation_results[i][sample_size].append(res)
                    progress_bar.update(1)

    # with open(f"{save_dir}/type1_pg_rates.pkl", "wb") as out_file:
    #    pickle.dump(simulation_results, out_file)

    # Plot Type I error rates as line plot
    plt.figure(figsize=(8, 6))
    plt.rcParams.update(
        {"font.size": 18, "text.usetex": True, "legend.loc": "upper right"}
    )

    for test_name, data in simulation_results.items():
        color, marker, marker_size = None, None, None

        if colors_and_markers is not None:
            color, marker = colors_and_markers[test_name]
            marker_size = 16

        y = [
            (threshold >= np.array(data[sample_size]).astype(float).mean())
            + (1 - threshold <= np.array(data[sample_size]).astype(float).mean())
            for sample_size in sample_sizes
        ]
        plt.plot(
            sample_sizes,
            y,
            label=test_name,
            color=color,
            marker=marker,
            markersize=marker_size,
            alpha=0.8,
        )

    ax = plt.gca()
    # ax.set_ylim(0, 1)
    ax.yaxis.grid()
    plt.xticks(sample_sizes, [str(size) for size in sample_sizes])
    plt.xlabel("Sample Size")
    plt.ylabel("Type I Error Rate")
    plt.legend()

    if save_dir is not None:
        plt.tight_layout()
        plt.savefig(f"{save_dir}/type1_pg_rates2.png")
    else:
        plt.show()

    plt.close()

    # Plot box-and-whiskers plot of values
    plt.figure(figsize=(8, 6))
    plt.rcParams.update(
        {"font.size": 20, "text.usetex": True, "legend.loc": "upper right"}
    )

    # Create datastructure for boxplots
    data = [
        [simulation_results[test_name][size] for size in sample_sizes]
        for test_name in range(4)
    ]

    # Create offsets for box plots
    spacing = 0.5
    offsets = np.arange(0, spacing * 3, spacing) - spacing * (3 - 1) / 2

    for test_name, test_data, offset in zip(range(4), data, offsets):
        color, marker = (
            (None, None)
            if colors_and_markers is None
            else colors_and_markers[test_name]
        )

        box_plot = plt.boxplot(
            test_data,
            positions=np.arange(0, len(sample_sizes)) * 3 + offset,
            sym=marker,
            widths=0.45,
            flierprops={"marker": marker},
        )

        if color is not None:
            plt.setp(box_plot["boxes"], color=color)
            plt.setp(box_plot["whiskers"], color=color)
            plt.setp(box_plot["caps"], color=color)
            plt.setp(box_plot["medians"], color=color)

            plt.plot([], color=color, label=test_name)

    ax = plt.gca()
    ax.set_ylim(0, 1)
    ax.set_xlim(-2, len(sample_sizes) * 3)
    ax.yaxis.grid()
    plt.xticks(np.arange(0, len(sample_sizes) * 3, 3), sample_sizes)
    plt.xlabel("Sample Size")
    plt.ylabel(r"$p$-value / $\varepsilon_\mathrm{min}$")
    plt.legend()

    if save_dir is not None:
        plt.tight_layout()
        plt.savefig(f"{save_dir}/type1_pg_dists2.png")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    test_type1_error(
        sample_sizes=SAMPLE_SIZES,
        save_dir=SAVE_DIR,
        num_simulations=NUM_SIMULATIONS,
    )

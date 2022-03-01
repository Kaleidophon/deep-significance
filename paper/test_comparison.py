"""
Compare ASO against other significance tests and measure Type I and Type II error under an increasing number of samples.
"""

# STD
import os
import pickle
from typing import Dict, Callable, List, Tuple, Optional, Any

# EXT
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
from tqdm import tqdm

# PACKAGE
from deepsig import aso, bootstrap_test, permutation_test

# CONST
CONSIDERED_TESTS = {
    "ASO": lambda a, b: aso(a, b, show_progress=False, num_jobs=4),
    "Student's t": lambda a, b: ttest_ind(a, b, equal_var=False, alternative="greater")[
        1
    ],
    "Bootstrap": bootstrap_test,
    "Permutation": permutation_test,
}
CONSIDERED_TEST_COLORS_MARKERS = {
    "ASO": ("darkred", "*"),
    "Student's t": ("darkblue", "o"),
    "Bootstrap": ("forestgreen", "^"),
    "Permutation": ("darkorange", "P"),
}
SAMPLE_SIZES = [5, 10]  # TODO: Debug , 15, 20, 25]
MEAN_DIFFS = [0.25, 0.5]  # TODO: Debug , 0.75, 1]
SAVE_DIR = "./img"
NUM_SIMULATIONS = 2  # TODO: Debug 250


def test_type1_error(
    tests: Dict[str, Callable],
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
        for test_name in tests
    }

    with tqdm(total=len(sample_sizes) * num_simulations * len(tests)) as progress_bar:
        for sample_size in sample_sizes:
            for _ in range(num_simulations):

                # Sample scores for this round
                scores_a = dist_func(**dist_params, size=sample_size)
                scores_b = dist_func(**dist_params, size=sample_size)

                for test_name, test_func in tests.items():
                    simulation_results[test_name][sample_size].append(
                        test_func(scores_a, scores_b)
                    )
                    progress_bar.update(1)

    with open(f"{save_dir}/type1_rates.pkl", "wb") as out_file:
        pickle.dump(simulation_results, out_file)

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
            (np.array(data[sample_size]) <= threshold).astype(float).mean()
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
        plt.savefig(f"{save_dir}/type1_rates.png")
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
        for test_name in tests
    ]

    # Create offsets for box plots
    spacing = 0.5
    offsets = (
        np.arange(0, spacing * len(tests), spacing) - spacing * (len(tests) - 1) / 2
    )

    for test_name, test_data, offset in zip(tests.keys(), data, offsets):
        color, marker = (
            (None, None)
            if colors_and_markers is None
            else colors_and_markers[test_name]
        )

        box_plot = plt.boxplot(
            test_data,
            positions=np.arange(0, len(sample_sizes)) * len(tests) + offset,
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
    ax.set_xlim(-2, len(sample_sizes) * len(tests))
    ax.yaxis.grid()
    plt.xticks(np.arange(0, len(sample_sizes) * len(tests), len(tests)), sample_sizes)
    plt.xlabel("Sample Size")
    plt.ylabel("Test value")
    plt.legend()

    if save_dir is not None:
        plt.tight_layout()
        plt.savefig(f"{save_dir}/type1_dists.png")
    else:
        plt.show()

    plt.close()


def test_type2_error_sample_size(
    tests: Dict[str, Callable],
    sample_sizes: List[int],
    num_simulations: int = 200,
    dist_func: Callable = np.random.normal,
    dist1_params: Dict[str, Any] = {"loc": 0.5, "scale": 1.5},
    dist2_params: Dict[str, Any] = {"loc": 0, "scale": 1.5},
    threshold: float = 0.05,
    colors_and_markers: Optional[Dict[str, Tuple[str, str]]] = None,
    save_dir: Optional[str] = None,
):
    """
    Test the rate of type 2 error (false negative) under different samples sizes.

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
    dist1_params: Dict[str, Any]
        Parameters of the first distribution function.
    dist2_params: Dict[str, Any]
        Parameters of the second distribution function.
    threshold: float
        Threshold that test results has to fall below in order for significance to be claimed.
    colors_and_markers: Optional[Dict[str, Tuple[str, str]]]
        Colors and markers corresponding to each test for plotting.
    save_dir: Optional[str]
        Directory that plots should be saved to.
    """
    simulation_results = {
        test_name: {sample_size: [] for sample_size in sample_sizes}
        for test_name in tests
    }

    with tqdm(total=len(sample_sizes) * num_simulations * len(tests)) as progress_bar:
        for sample_size in sample_sizes:
            for _ in range(num_simulations):

                # Sample scores for this round
                scores_a = dist_func(**dist1_params, size=sample_size)
                scores_b = dist_func(**dist2_params, size=sample_size)

                for test_name, test_func in tests.items():
                    simulation_results[test_name][sample_size].append(
                        test_func(scores_a, scores_b)
                    )
                    progress_bar.update(1)

    with open(f"{save_dir}/type2_rates.pkl", "wb") as out_file:
        pickle.dump(simulation_results, out_file)

    # Plot Type I error rates as line plot
    plt.figure(figsize=(8, 6))
    plt.rcParams.update(
        {"font.size": 20, "text.usetex": True, "legend.loc": "upper right"}
    )

    for test_name, data in simulation_results.items():
        color, marker, marker_size = None, None, None

        if colors_and_markers is not None:
            color, marker = colors_and_markers[test_name]
            marker_size = 16

        y = [
            1 - (np.array(data[sample_size]) <= threshold).astype(float).mean()
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
    plt.ylabel("Type II Error Rate")
    plt.legend()

    if save_dir is not None:
        plt.tight_layout()
        plt.savefig(f"{save_dir}/type2_rates.png")
    else:
        plt.show()

    plt.close()


def test_type2_error_mean_difference(
    tests: Dict[str, Callable],
    mean_differences: List[float],
    num_simulations: int = 200,
    target_param: str = "loc",
    dist_func: Callable = np.random.normal,
    dist_params: Dict[str, Any] = {"loc": 0, "scale": 1.5},
    sample_size: int = 5,
    threshold: float = 0.05,
    colors_and_markers: Optional[Dict[str, Tuple[str, str]]] = None,
    save_dir: Optional[str] = None,
):
    """
    Test the rate of type II error under different mean differences between the two distributions that samples are taken
    from.

    Parameters
    ----------
    tests: Dict[str, Callable]
        Considered tests.
    mean_differences: List[float]
        Mean differences between distributions that simulations are run for.
    num_simulations: int
        Number of simulations conducted.
    target_param: str
        Name of parameter affected by mean_differences.
    dist_func: Callable
        Distribution function that is used for sampling.
    dist_params: Dict[str, Any]
        Parameters of the distribution function.
    sample_size: int
        Number of samples for simulations.
    threshold: float
        Threshold that test results has to fall below in order for significance to be claimed.
    colors_and_markers: Optional[Dict[str, Tuple[str, str]]]
        Colors and markers corresponding to each test for plotting.
    save_dir: Optional[str]
        Directory that plots should be saved to.
    """
    simulation_results = {
        test_name: {mean_diff: [] for mean_diff in mean_differences}
        for test_name in tests
    }

    with tqdm(
        total=len(mean_differences) * num_simulations * len(tests)
    ) as progress_bar:
        for mean_diff in mean_differences:
            for _ in range(num_simulations):

                # Sample scores for this round
                modified_dist_params = {
                    param: (value + mean_diff if param == target_param else value)
                    for param, value in dist_params.items()
                }
                scores_a = dist_func(**modified_dist_params, size=sample_size)
                scores_b = dist_func(**dist_params, size=sample_size)

                for test_name, test_func in tests.items():
                    simulation_results[test_name][mean_diff].append(
                        test_func(scores_a, scores_b)
                    )
                    progress_bar.update(1)

    with open(f"{save_dir}/type2_mean_rates.pkl", "wb") as out_file:
        pickle.dump(simulation_results, out_file)

    # Plot Type I error rates as line plot
    plt.figure(figsize=(8, 6))
    plt.rcParams.update(
        {"font.size": 20, "text.usetex": True, "legend.loc": "upper right"}
    )

    for test_name, data in simulation_results.items():
        color, marker, marker_size = None, None, None

        if colors_and_markers is not None:
            color, marker = colors_and_markers[test_name]
            marker_size = 16

        y = [
            1 - (np.array(data[sample_size]) <= threshold).astype(float).mean()
            for sample_size in mean_differences
        ]
        plt.plot(
            mean_differences,
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
    plt.xticks(mean_differences, [str(size) for size in mean_differences])
    plt.xlabel("Mean difference")
    plt.ylabel("Type II Error Rate")
    plt.legend()

    if save_dir is not None:
        plt.tight_layout()
        plt.savefig(f"{save_dir}/type2_mean_rates.png")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    for dist_func, target_param, dist1_params, dist2_params, save_dir in zip(
        [np.random.normal, np.random.laplace, np.random.rayleigh],
        ["loc", "loc", "scale"],
        [{"loc": 0.5, "scale": 1.5}, {"loc": 0.5, "scale": 1.5}, {"scale": 1}],
        [{"loc": 0, "scale": 1.5}, {"loc": 0, "scale": 1.5}, {"scale": 0.5}],
        [f"{SAVE_DIR}/normal", f"{SAVE_DIR}/laplace", f"{SAVE_DIR}/rayleigh"],
    ):

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        test_type1_error(
            tests=CONSIDERED_TESTS,
            dist_func=dist_func,
            dist_params=dist2_params,
            sample_sizes=SAMPLE_SIZES,
            colors_and_markers=CONSIDERED_TEST_COLORS_MARKERS,
            save_dir=save_dir,
            num_simulations=NUM_SIMULATIONS,
        )

        if dist_func == np.random.normal:
            test_type2_error_sample_size(
                tests=CONSIDERED_TESTS,
                dist_func=dist_func,
                dist1_params=dist1_params,
                dist2_params=dist2_params,
                sample_sizes=SAMPLE_SIZES,
                colors_and_markers=CONSIDERED_TEST_COLORS_MARKERS,
                save_dir=save_dir,
                num_simulations=NUM_SIMULATIONS,
            )

            test_type2_error_mean_difference(
                tests=CONSIDERED_TESTS,
                dist_func=dist_func,
                dist_params=dist2_params,
                target_param=target_param,
                mean_differences=MEAN_DIFFS,
                colors_and_markers=CONSIDERED_TEST_COLORS_MARKERS,
                save_dir=save_dir,
                num_simulations=NUM_SIMULATIONS,
            )

"""
Compare ASO against other significance tests and measure Type I and Type II error under an increasing number of samples.
"""

# STD
from typing import Dict, Callable, List, Tuple, Optional

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
SAMPLE_SIZES = [5, 10, 15, 25, 40, 50]
SAVE_DIR = "./img"


def test_type1_error(
    tests: Dict[str, Callable],
    sample_sizes: List[int],
    num_simulations: int = 50,
    loc: float = 0,
    scale: float = 1,
    threshold: float = 0.05,
    colors_and_markers: Optional[Dict[str, Tuple[str, str]]] = None,
    save_dir: Optional[str] = None,
):
    """
    Test the rate of type I error under different samples sizes.

    Parameters
    ----------
    tests: Dict[str, Callable]
        Considered tests.
    sample_sizes: List[int]
        Samples sizes that are being tested.
    num_simulations: int
        Number of simulations conducted.
    loc: float
        Location of the normal distribution both samples are taken from.
    scale: float
        Scale of the normal distribution both samples are taken from.
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
                scores_a = np.random.normal(loc=loc, scale=scale, size=sample_size)
                scores_b = np.random.normal(loc=loc, scale=scale, size=sample_size)

                for test_name, test_func in tests.items():
                    simulation_results[test_name][sample_size].append(
                        test_func(scores_a, scores_b)
                    )
                    progress_bar.update(1)

    # Plot Type I error rates as line plot
    plt.figure(figsize=(8, 6))
    plt.rcParams.update(
        {
            "font.size": 18,
            "text.usetex": True,
        }
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
        {
            "font.size": 18,
            "text.usetex": True,
        }
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
            sym="",
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
    num_simulations=1000,
    loc1: float = 0.2,
    scale1: float = 1,
    loc2: float = 0,
    scale2: float = 1,
):
    """
    Test the rate of type II error under different samples sizes.

    Parameters
    ----------
    tests: Dict[str, Callable]
        Considered tests.
    sample_sizes: List[int]
        Samples sizes that are being tested.
    num_simulations: int
        Number of simulations conducted.
    loc1: float
        Location of the first normal distribution.
    scale1: float
        Scale of the first normal distribution.
    loc2: float
        Location of second normal distribution.
    scale2: float
        Scale of the second normal distribution.
    """
    ...  # TODO


def test_type2_error_mean_difference(
    tests: Dict[str, Callable],
    mean_differences: List[float],
    num_simulations=1000,
    loc: float = 0,
    scale: float = 1,
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
    loc: float
        Location of second normal distribution.
    scale: float
        Scale for both normal distributions.
    """
    ...  # TODO


if __name__ == "__main__":
    test_type1_error(
        tests=CONSIDERED_TESTS,
        sample_sizes=SAMPLE_SIZES,
        colors_and_markers=CONSIDERED_TEST_COLORS_MARKERS,
        save_dir=SAVE_DIR,
    )

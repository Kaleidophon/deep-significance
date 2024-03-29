"""
Compare ASO against other significance tests and measure Type I and Type II error under an increasing number of samples.
"""

# STD
import argparse
from itertools import product
import os
import pickle
from typing import Dict, Callable, List, Tuple, Optional, Any
import warnings

# EXT
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu
import pandas as pd
from tqdm import tqdm

# PACKAGE
from deepsig import aso, bootstrap_test, permutation_test

# CONST
NUM_JOBS_ASO = 16
NUM_JOBS_REST = 4
CONSIDERED_TESTS = {
    "ASO": lambda a, b: aso(a, b, show_progress=False, num_jobs=NUM_JOBS_ASO),
    "Student's t": lambda a, b: ttest_ind(a, b, equal_var=False, alternative="greater")[
        1
    ],
    "Bootstrap": lambda a, b: bootstrap_test(a, b, num_jobs=NUM_JOBS_REST),
    "Permutation": lambda a, b: permutation_test(a, b, num_jobs=NUM_JOBS_REST),
    "Wilcoxon": lambda a, b: wilcoxon(a, b, alternative="greater").pvalue,
    "Mann-Whitney U": lambda a, b: mannwhitneyu(a, b, alternative="greater").pvalue,
}
CONSIDERED_TEST_COLORS_MARKERS = {
    "ASO": ("darkred", "*"),
    "Student's t": ("darkblue", "o"),
    "Bootstrap": ("forestgreen", "^"),
    "Permutation": ("darkorange", "P"),
    "Wilcoxon": ("darkviolet", "x"),
    "Mann-Whitney U": ("slategray", "p"),
}
SAMPLE_SIZES = [5, 10, 15, 20]
MEAN_DIFFS = [0.25, 0.5, 0.75, 1]
SAVE_DIR = "./img"
NUM_SIMULATIONS = {
    "ASO": 500,
    "Student's t": 1000,
    "Bootstrap": 1000,
    "Permutation": 1000,
    "Wilcoxon": 1000,
    "Mann-Whitney U": 1000,
}
P_VALUE_TRESHOLD = 0.05
ASO_THRESHOLD = 0.2
ALL_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]


def test_type1_error(
    tests: Dict[str, Callable],
    sample_sizes: List[int],
    num_simulations: Dict[str, int],
    dist_func: Callable = np.random.normal,
    dist_params: Dict[str, Any] = {"loc": 0, "scale": 1.5},
    p_value_threshold: float = 0.05,
    aso_threshold: float = 0.2,
    all_thresholds: List[float] = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    y_lim: Optional[Tuple[float, float]] = None,
    colors_and_markers: Optional[Dict[str, Tuple[str, str]]] = None,
    save_dir: Optional[str] = None,
    plot: bool = True,
    plot_from_pickle: bool = False,
):
    """
    Test the rate of type I error (false positive) under different samples sizes.

    Parameters
    ----------
    tests: Dict[str, Callable]
        Considered tests.
    sample_sizes: List[int]
        Samples sizes that are being tested.
    num_simulations: Dict[str, int]
        Number of simulations conducted per method as dict.
    dist_func: Callable
        Distribution function that is used for sampling.
    dist_params: Dict[str, Any]
        Parameters of the distribution function.
    p_value_threshold: float
        Threshold that test results has to fall below in order for significance to be claimed.
    aso_threshold: float
        Threshold that ASO test results has to fall below in order for significance to be claimed.
    all_thresholds: List[float]
        List of all threshold that Type I error will be computed and printed as a table for.
    colors_and_markers: Optional[Dict[str, Tuple[str, str]]]
        Colors and markers corresponding to each test for plotting.
    save_dir: Optional[str]
        Directory that plots should be saved to.
    plot: bool
        Indicate whether plots should be created.
    plot_from_pickle: bool
        Indicate whether simulating experiments should be skipped in favor of just loading results from a pickle.
        Default is false.
    """

    if not plot_from_pickle:
        simulation_results = {
            test_name: {sample_size: [] for sample_size in sample_sizes}
            for test_name in tests
        }
        max_simulations = max(num_simulations.values())

        with tqdm(
            total=len(sample_sizes) * sum(num_simulations.values())
        ) as progress_bar:
            for sample_size in sample_sizes:
                for simulation_idx in range(max_simulations):

                    # Sample scores for this round
                    scores_a = dist_func(**dist_params, size=sample_size)
                    scores_b = dist_func(**dist_params, size=sample_size)

                    for test_name, test_func in tests.items():

                        if simulation_idx < num_simulations[test_name]:
                            simulation_results[test_name][sample_size].append(
                                test_func(scores_a, scores_b)
                            )
                            progress_bar.update(1)

        with open(f"{save_dir}/type1_rates.pkl", "wb") as out_file:
            pickle.dump(simulation_results, out_file)

    else:
        try:
            with open(f"{save_dir}/type1_rates.pkl", "rb") as in_file:
                simulation_results = pickle.load(in_file)

            # Overwrite with sample_sizes actually found in loaded pickle
            first_key = list(simulation_results.keys())[0]
            sample_sizes = list(simulation_results[first_key].keys())

        except FileNotFoundError:
            warnings.warn(
                f"File '{save_dir}/type1_rates.pkl' not found, no plots generated."
            )
            return

    # Plotting
    if plot or plot_from_pickle:
        y = {
            test_name: [
                (
                    np.array(simulation_results[test_name][sample_size])
                    <= (p_value_threshold if test_name != "ASO" else aso_threshold)
                )
                .astype(float)
                .mean()
                for sample_size in sample_sizes
            ]
            for test_name in tests
        }
        plot_lines(
            y=y,
            groups=sample_sizes,
            x_label="Sample Size",
            y_label="Type I Error Rate",
            save_dir=save_dir,
            file_name="type1_rates",
            y_lim=y_lim,
            colors_and_markers=colors_and_markers,
        )

        plot_boxes(
            results=simulation_results,
            tests=tests,
            groups=sample_sizes,
            x_label="Sample Size",
            y_label=r"$p$-value / $\varepsilon_\mathrm{min}$",
            save_dir=save_dir,
            file_name="type1_dists",
            colors_and_markers=colors_and_markers,
        )

    # Creating a table
    df = pd.DataFrame(
        columns=tests.keys(),
        index=pd.MultiIndex.from_tuples(
            product(sample_sizes, all_thresholds, repeat=1),
            names=["sample_size", "threshold"],
        ),
    )

    for sample_size in sample_sizes:
        for threshold in all_thresholds:
            for test_name in tests:
                df.at[(sample_size, threshold), test_name] = (
                    (
                        np.array(simulation_results[test_name][sample_size])
                        <= (threshold if test_name != "ASO" else threshold)
                    )
                    .astype(float)
                    .mean()
                )

    print(df.to_latex())


def test_type2_error_sample_size(
    tests: Dict[str, Callable],
    sample_sizes: List[int],
    num_simulations: Dict[str, int],
    dist_func: Callable = np.random.normal,
    dist1_params: Dict[str, Any] = {"loc": 0.5, "scale": 1.5},
    dist2_params: Dict[str, Any] = {"loc": 0, "scale": 1.5},
    p_value_threshold: float = 0.05,
    aso_threshold: float = 0.2,
    all_thresholds: List[float] = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    y_lim: Optional[Tuple[float, float]] = None,
    colors_and_markers: Optional[Dict[str, Tuple[str, str]]] = None,
    save_dir: Optional[str] = None,
    plot: bool = True,
    plot_from_pickle: bool = False,
):
    """
    Test the rate of type 2 error (false negative) under different samples sizes.

    Parameters
    ----------
    tests: Dict[str, Callable]
        Considered tests.
    sample_sizes: List[int]
        Samples sizes that are being tested.
    num_simulations: Dict[str, int]
        Number of simulations conducted per method as dict.
    dist_func: Callable
        Distribution function that is used for sampling.
    dist1_params: Dict[str, Any]
        Parameters of the first distribution function.
    dist2_params: Dict[str, Any]
        Parameters of the second distribution function.
    p_value_threshold: float
        Threshold that test results has to fall below in order for significance to be claimed.
    aso_threshold: float
        Threshold that ASO test results has to fall below in order for significance to be claimed.
    all_thresholds: List[float]
        List of all threshold that Type I error will be computed and printed as a table for.
    colors_and_markers: Optional[Dict[str, Tuple[str, str]]]
        Colors and markers corresponding to each test for plotting.
    save_dir: Optional[str]
        Directory that plots should be saved to.
    plot: bool
        Indicate whether plots should be created.
    plot_from_pickle: bool
        Indicate whether simulating experiments should be skipped in favor of just loading results from a pickle.
        Default is false.
    """
    if not plot_from_pickle:
        simulation_results = {
            test_name: {sample_size: [] for sample_size in sample_sizes}
            for test_name in tests
        }
        max_simulations = max(num_simulations.values())

        with tqdm(
            total=len(sample_sizes) * sum(num_simulations.values())
        ) as progress_bar:
            for sample_size in sample_sizes:
                for simulation_idx in range(max_simulations):

                    # Sample scores for this round
                    scores_a = dist_func(**dist1_params, size=sample_size)
                    scores_b = dist_func(**dist2_params, size=sample_size)

                    for test_name, test_func in tests.items():

                        if simulation_idx < num_simulations[test_name]:
                            simulation_results[test_name][sample_size].append(
                                test_func(scores_a, scores_b)
                            )
                            progress_bar.update(1)

        with open(f"{save_dir}/type2_rates.pkl", "wb") as out_file:
            pickle.dump(simulation_results, out_file)

    else:
        try:
            with open(f"{save_dir}/type2_rates.pkl", "rb") as in_file:
                simulation_results = pickle.load(in_file)

            # Overwrite with sample_sizes actually found in loaded pickle
            first_key = list(simulation_results.keys())[0]
            sample_sizes = list(simulation_results[first_key].keys())

        except FileNotFoundError:
            warnings.warn(
                f"File '{save_dir}/type2_rates.pkl' not found, no plots generated."
            )
            return

    # Plot Type II error rates as line plot
    if plot or plot_from_pickle:
        y = {
            test_name: [
                (
                    np.array(simulation_results[test_name][sample_size])
                    >= (p_value_threshold if test_name != "ASO" else aso_threshold)
                )
                .astype(float)
                .mean()
                for sample_size in sample_sizes
            ]
            for test_name in tests
        }
        plot_lines(
            y=y,
            groups=sample_sizes,
            x_label="Sample Size",
            y_label="Type II Error Rate",
            save_dir=save_dir,
            y_lim=y_lim,
            file_name="type2_rates",
            colors_and_markers=colors_and_markers,
        )

        plot_boxes(
            results=simulation_results,
            tests=tests,
            groups=sample_sizes,
            x_label="Sample Size",
            y_label=r"$p$-value / $\varepsilon_\mathrm{min}$",
            save_dir=save_dir,
            file_name="type2_dists",
            colors_and_markers=colors_and_markers,
        )

    # Creating a table
    df = pd.DataFrame(
        columns=tests.keys(),
        index=pd.MultiIndex.from_tuples(
            product(sample_sizes, all_thresholds, repeat=1),
            names=["sample_size", "threshold"],
        ),
    )

    for sample_size in sample_sizes:
        for threshold in all_thresholds:
            for test_name in tests:
                df.at[(sample_size, threshold), test_name] = (
                    (np.array(simulation_results[test_name][sample_size]) >= threshold)
                    .astype(float)
                    .mean()
                )

    print(df.to_latex())


def test_type2_error_mean_difference(
    tests: Dict[str, Callable],
    mean_differences: List[float],
    num_simulations: Dict[str, int],
    target_param: str = "loc",
    dist_func: Callable = np.random.normal,
    dist_params: Dict[str, Any] = {"loc": 0, "scale": 1.5},
    sample_size: int = 5,
    p_value_threshold: float = 0.05,
    aso_threshold: float = 0.2,
    all_thresholds: List[float] = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    y_lim: Optional[Tuple[float, float]] = None,
    colors_and_markers: Optional[Dict[str, Tuple[str, str]]] = None,
    save_dir: Optional[str] = None,
    plot: bool = True,
    plot_from_pickle: bool = False,
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
    num_simulations: Dict[str, int]
        Number of simulations conducted per method as dict.
    target_param: str
        Name of parameter affected by mean_differences.
    dist_func: Callable
        Distribution function that is used for sampling.
    dist_params: Dict[str, Any]
        Parameters of the distribution function.
    sample_size: int
        Number of samples for simulations.
    p_value_threshold: float
        Threshold that test results has to fall below in order for significance to be claimed.
    aso_threshold: float
        Threshold that ASO test results has to fall below in order for significance to be claimed.
    all_thresholds: List[float]
        List of all threshold that Type I error will be computed and printed as a table for.
    colors_and_markers: Optional[Dict[str, Tuple[str, str]]]
        Colors and markers corresponding to each test for plotting.
    save_dir: Optional[str]
        Directory that plots should be saved to.
    plot: bool
        Indicate whether plots should be created.
    plot_from_pickle: bool
        Indicate whether simulating experiments should be skipped in favor of just loading results from a pickle.
        Default is false.
    """
    if not plot_from_pickle:
        simulation_results = {
            test_name: {mean_diff: [] for mean_diff in mean_differences}
            for test_name in tests
        }
        max_simulations = max(num_simulations.values())

        with tqdm(
            total=len(mean_differences) * sum(num_simulations.values())
        ) as progress_bar:
            for mean_diff in mean_differences:
                for simulation_idx in range(max_simulations):

                    # Sample scores for this round
                    modified_dist_params = {
                        param: (value + mean_diff if param == target_param else value)
                        for param, value in dist_params.items()
                    }
                    scores_a = dist_func(**modified_dist_params, size=sample_size)
                    scores_b = dist_func(**dist_params, size=sample_size)

                    for test_name, test_func in tests.items():

                        if simulation_idx < num_simulations[test_name]:
                            simulation_results[test_name][mean_diff].append(
                                test_func(scores_a, scores_b)
                            )
                            progress_bar.update(1)

        with open(f"{save_dir}/type2_mean_rates.pkl", "wb") as out_file:
            pickle.dump(simulation_results, out_file)

    else:
        try:
            with open(f"{save_dir}/type2_mean_rates.pkl", "rb") as in_file:
                simulation_results = pickle.load(in_file)

            # Overwrite with sample_sizes actually found in loaded pickle
            first_key = list(simulation_results.keys())[0]
            mean_differences = list(simulation_results[first_key].keys())

        except FileNotFoundError:
            warnings.warn(
                f"File '{save_dir}/type2_mean_rates.pkl' not found, no plots generated."
            )
            return

    # Plot Type II error rates as line plot
    if plot or plot_from_pickle:
        y = {
            test_name: [
                (
                    np.array(simulation_results[test_name][mean_difference])
                    >= (p_value_threshold if test_name != "ASO" else aso_threshold)
                )
                .astype(float)
                .mean()
                for mean_difference in mean_differences
            ]
            for test_name in tests
        }
        plot_lines(
            y=y,
            groups=mean_differences,
            x_label="Mean difference",
            y_label="Type II Error Rate",
            save_dir=save_dir,
            y_lim=y_lim,
            file_name="type2_mean_rates",
            colors_and_markers=colors_and_markers,
        )

        plot_boxes(
            results=simulation_results,
            tests=tests,
            groups=mean_differences,
            x_label="Mean difference",
            y_label=r"$p$-value / $\varepsilon_\mathrm{min}$",
            save_dir=save_dir,
            file_name="type2_mean_dists",
            colors_and_markers=colors_and_markers,
        )

    # Creating a table
    df = pd.DataFrame(
        columns=tests.keys(),
        index=pd.MultiIndex.from_tuples(
            product(mean_differences, all_thresholds, repeat=1),
            names=["diff", "threshold"],
        ),
    )

    for diff in mean_differences:
        for threshold in all_thresholds:
            for test_name in tests:
                df.at[(diff, threshold), test_name] = (
                    (np.array(simulation_results[test_name][diff]) >= threshold)
                    .astype(float)
                    .mean()
                )

    print(df.to_latex())


def plot_lines(
    y: Dict[str, List[float]],
    groups: List[Any],
    x_label: str,
    y_label: str,
    save_dir: str,
    file_name: str,
    y_lim: Optional[Tuple[float, float]] = None,
    colors_and_markers: Optional[Dict[str, Tuple[str, str]]] = None,
):
    """
    Plot data as line plots.

    Parameters
    ----------
    y: Dict[str, List[float]]
        Data to be plotted as data by test by group.
    groups: List[Any]
        Names of groups to be plotted on the x-axis.
    x_label: str
        x-axis label.
    y_label: str
        y-axis label.
    save_dir: str
        Directory the plot should be saved to.
    file_name: str
        File name for the plot.
    colors_and_markers: Optional[Dict[str, Tuple[str, str]]]
        Colors and markers corresponding to each test for plotting.
    """
    # Plot Type I error rates as line plot
    plt.figure(figsize=(8, 6))
    plt.rcParams.update(
        {"font.size": 18, "text.usetex": True, "legend.loc": "upper right"}
    )

    for test_name, data in y.items():
        color, marker, marker_size = None, None, None

        if colors_and_markers is not None:
            color, marker = colors_and_markers[test_name]
            marker_size = 16

        plt.plot(
            groups,
            data,
            label=test_name,
            color=color,
            marker=marker,
            markersize=marker_size,
            alpha=0.8,
        )

    ax = plt.gca()

    if y_lim is not None:
        ax.set_ylim(*y_lim)

    ax.yaxis.grid()
    plt.xticks(groups, [str(group) for group in groups])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{file_name}.png")
    plt.close()


def plot_boxes(
    tests: Dict[str, Callable],
    results: Dict[str, Dict[str, float]],
    groups: List[Any],
    x_label: str,
    y_label: str,
    save_dir: str,
    file_name: str,
    colors_and_markers: Optional[Dict[str, Tuple[str, str]]] = None,
):
    """
    Plot data as box-and-whiskers plot.

    Parameters
    ----------
    tests: Dict[str, Callable]
        Considered tests.
    results: Dict[str, Dict[str, float]]
        Simulation results.
    groups: List[Any]
        Names of groups to be plotted on the x-axis.
    x_label: str
        x-axis label.
    y_label: str
        y-axis label.
    save_dir: str
        Directory the plot should be saved to.
    file_name: str
        File name for the plot.
    colors_and_markers: Optional[Dict[str, Tuple[str, str]]]
        Colors and markers corresponding to each test for plotting.
    """
    # Create data structure for boxplots
    data = [[results[test_name][group] for group in groups] for test_name in tests]

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
            positions=np.arange(0, len(groups)) * len(tests) + offset,
            sym=marker,
            widths=0.45,
            flierprops={"marker": ".", "markerfacecolor": color},
        )

        if color is not None:
            plt.setp(box_plot["boxes"], color=color)
            plt.setp(box_plot["whiskers"], color=color)
            plt.setp(box_plot["caps"], color=color)
            plt.setp(box_plot["medians"], color=color)

            plt.plot([], color=color, label=test_name)

    ax = plt.gca()
    ax.set_ylim(0, 1)
    ax.set_xlim(-2, len(groups) * len(tests))
    ax.yaxis.grid()
    plt.xticks(np.arange(0, len(groups) * len(tests), len(tests)), groups)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{file_name}.png")

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-from-pickle", action="store_true", default=False)
    parser.add_argument("--plot", action="store_true", default=False)
    args = parser.parse_args()

    def normal_mixture(
        loc: float,
        scale: float,
        size: int,
        loc2: float = -0.5,
        scale2: float = 0.25,
        mixture_coeff: float = 0.7,
    ):
        """
        Define a simple sampling procedure from a mixture of two normal distributions.
        """
        samples1 = np.random.normal(
            loc=loc, scale=scale, size=int(size * mixture_coeff)
        )
        samples2 = np.random.normal(
            loc=loc2, scale=scale2, size=size - int(size * mixture_coeff)
        )

        combined_samples = np.concatenate((samples1, samples2))

        return combined_samples

    for dist_func, target_param, dist1_params, dist2_params, save_dir in zip(
        [np.random.normal, normal_mixture, np.random.laplace, np.random.rayleigh],
        ["loc", "loc", "loc", "scale"],
        [
            {"loc": 0.5, "scale": 1.5},
            {"loc": 2.5, "scale": 1},
            {"loc": 0.5, "scale": 1.5},
            {"scale": 2},
        ],
        [
            {"loc": 0, "scale": 1.5},
            {"loc": 1.5, "scale": 1},
            {"loc": 0, "scale": 1.5},
            {"scale": 1},
        ],
        [
            f"{SAVE_DIR}/normal",
            f"{SAVE_DIR}/normal_mix",
            f"{SAVE_DIR}/laplace",
            f"{SAVE_DIR}/rayleigh",
        ],
    ):
        print(f"{'#' * 10} {dist_func.__name__} {'#' * 10}")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        print("Type I error")
        test_type1_error(
            tests=CONSIDERED_TESTS,
            dist_func=dist_func,
            dist_params=dist2_params,
            sample_sizes=SAMPLE_SIZES,
            p_value_threshold=P_VALUE_TRESHOLD,
            aso_threshold=ASO_THRESHOLD,
            all_thresholds=ALL_THRESHOLDS,
            colors_and_markers=CONSIDERED_TEST_COLORS_MARKERS,
            y_lim=(0, 0.12),
            save_dir=save_dir,
            num_simulations=NUM_SIMULATIONS,
            plot=args.plot,
            plot_from_pickle=args.plot_from_pickle,
        )

        if dist_func in (np.random.normal, normal_mixture):
            print("Type II error sample size")
            test_type2_error_sample_size(
                tests=CONSIDERED_TESTS,
                dist_func=dist_func,
                dist1_params=dist1_params,
                dist2_params=dist2_params,
                sample_sizes=SAMPLE_SIZES,
                p_value_threshold=P_VALUE_TRESHOLD,
                aso_threshold=ASO_THRESHOLD,
                all_thresholds=ALL_THRESHOLDS,
                colors_and_markers=CONSIDERED_TEST_COLORS_MARKERS,
                save_dir=save_dir,
                num_simulations=NUM_SIMULATIONS,
                plot=args.plot,
                plot_from_pickle=args.plot_from_pickle,
            )

            print("Type II error mean differences")
            test_type2_error_mean_difference(
                tests=CONSIDERED_TESTS,
                dist_func=dist_func,
                dist_params=dist2_params,
                target_param=target_param,
                mean_differences=MEAN_DIFFS,
                p_value_threshold=P_VALUE_TRESHOLD,
                aso_threshold=ASO_THRESHOLD,
                all_thresholds=ALL_THRESHOLDS,
                colors_and_markers=CONSIDERED_TEST_COLORS_MARKERS,
                save_dir=save_dir,
                num_simulations=NUM_SIMULATIONS,
                plot=args.plot,
                plot_from_pickle=args.plot_from_pickle,
            )

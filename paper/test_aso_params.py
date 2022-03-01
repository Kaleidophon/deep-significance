"""
Test the Type I error rate of ASO as a function of the confidence score and bootstrap iterations.
"""

# STD
import pickle
from typing import Optional, Tuple, List

# EXT
import matplotlib.pyplot as plt
from deepsig import aso
import numpy as np
from tqdm import tqdm

# CONST
COLOR_MARKER = ("darkred", "*")
SAVE_DIR = "./img"
NUM_SIMULATIONS = 250
BOOTSTRAP_ITERS = [250, 500, 750, 1000]
CONFIDENCE_LEVELS = [0.1, 0.05, 0.01, 0.005]


def test_type1_error_bootstrap(
    bootstrap_iters: List[int],
    num_simulations: int = 200,
    sample_size: int = 5,
    loc: float = 0,
    scale: float = 1.5,
    color_and_marker: Optional[Tuple[str, str]] = None,
    save_dir: Optional[str] = None,
):
    simulation_results = {iters: [] for iters in bootstrap_iters}

    with tqdm(total=len(bootstrap_iters) * num_simulations) as progress_bar:
        for iters in bootstrap_iters:
            for _ in range(num_simulations):

                # Sample scores for this round
                scores_a = np.random.normal(loc=loc, scale=scale, size=sample_size)
                scores_b = np.random.normal(loc=loc, scale=scale, size=sample_size)

                simulation_results[iters].append(
                    aso(
                        scores_a,
                        scores_b,
                        show_progress=False,
                        num_jobs=4,
                        num_bootstrap_iterations=iters,
                    )
                )
                progress_bar.update(1)

    with open(f"{save_dir}/type1_bootstrap_rates.pkl", "wb") as out_file:
        pickle.dump(simulation_results, out_file)

    # Plot Type I error rates as line plot
    plt.figure(figsize=(8, 6))
    plt.rcParams.update(
        {"font.size": 18, "text.usetex": True, "legend.loc": "upper right"}
    )

    color, marker = None, None

    if color_and_marker is not None:
        color, marker = color_and_marker

    box_plot = plt.boxplot(
        [simulation_results[iters] for iters in bootstrap_iters],
        sym=marker,
        widths=0.65,
        positions=np.arange(0, len(bootstrap_iters)),
        flierprops={"marker": marker},
    )

    if color is not None:
        plt.setp(box_plot["boxes"], color=color)
        plt.setp(box_plot["whiskers"], color=color)
        plt.setp(box_plot["caps"], color=color)
        plt.setp(box_plot["medians"], color=color)

    ax = plt.gca()
    # ax.set_ylim(0, 1)
    ax.yaxis.grid()
    plt.xticks(
        np.arange(0, len(bootstrap_iters)), [str(iters) for iters in bootstrap_iters]
    )
    plt.xlabel("Num. Bootstrap Iterations")
    plt.ylabel("Type I Error Rate")

    if save_dir is not None:
        plt.tight_layout()
        plt.savefig(f"{save_dir}/type1_bootstrap_rates.png")
    else:
        plt.show()

    plt.close()


def test_type1_error_confidence(
    confidence_levels: List[float],
    num_simulations: int = 200,
    sample_size: int = 5,
    loc: float = 0,
    scale: float = 1.5,
    color_and_marker: Optional[Tuple[str, str]] = None,
    save_dir: Optional[str] = None,
):
    simulation_results = {level: [] for level in confidence_levels}

    with tqdm(total=len(confidence_levels) * num_simulations) as progress_bar:
        for level in confidence_levels:
            for _ in range(num_simulations):
                # Sample scores for this round
                scores_a = np.random.normal(loc=loc, scale=scale, size=sample_size)
                scores_b = np.random.normal(loc=loc, scale=scale, size=sample_size)

                simulation_results[level].append(
                    aso(
                        scores_a,
                        scores_b,
                        show_progress=False,
                        num_jobs=4,
                        confidence_level=level,
                    )
                )
                progress_bar.update(1)

    with open(f"{save_dir}/type1_confidence_rates.pkl", "wb") as out_file:
        pickle.dump(simulation_results, out_file)

    # Plot Type I error rates as line plot
    plt.figure(figsize=(8, 6))
    plt.rcParams.update(
        {"font.size": 18, "text.usetex": True, "legend.loc": "upper right"}
    )

    color, marker = None, None

    if color_and_marker is not None:
        color, marker = color_and_marker

    box_plot = plt.boxplot(
        [simulation_results[level] for level in confidence_levels],
        positions=np.arange(0, len(confidence_levels)),
        sym=marker,
        widths=0.65,
        flierprops={"marker": marker},
    )

    if color is not None:
        plt.setp(box_plot["boxes"], color=color)
        plt.setp(box_plot["whiskers"], color=color)
        plt.setp(box_plot["caps"], color=color)
        plt.setp(box_plot["medians"], color=color)

    ax = plt.gca()
    # ax.set_ylim(0, 1)
    ax.yaxis.grid()
    plt.xticks(
        np.arange(0, len(confidence_levels)),
        [str(level) for level in confidence_levels],
    )
    plt.xlabel("Confidence level")
    plt.ylabel("Type I Error Rate")

    if save_dir is not None:
        plt.tight_layout()
        plt.savefig(f"{save_dir}/type1_confidence_rates.png")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    test_type1_error_bootstrap(
        BOOTSTRAP_ITERS,
        num_simulations=NUM_SIMULATIONS,
        color_and_marker=COLOR_MARKER,
        save_dir=SAVE_DIR,
    )

    test_type1_error_confidence(
        CONFIDENCE_LEVELS,
        num_simulations=NUM_SIMULATIONS,
        color_and_marker=COLOR_MARKER,
        save_dir=SAVE_DIR,
    )

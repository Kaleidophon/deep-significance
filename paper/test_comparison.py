"""
Compare ASO against other significance tests and measure Type I and Type II error under an increasing number of samples.
"""

# STD
from typing import Dict, Callable, List

# EXT
from scipy.stats import ttest_ind
from tqdm import tqdm

# PACKAGE
from deepsig import aso, bootstrap_test, permutation

# CONST
CONSIDERED_TEST = {
    "aso": aso,
    "t-test": lambda a, b: ttest_ind(a, b, equal_var=False),
    "bootstrap": bootstrap_test,
    "permutation": permutation,
}


def test_type1_error(
    tests: Dict[str, Callable],
    sample_sizes: List[int],
    num_simulations=1000,
    loc: float = 0,
    scale: float = 1,
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
    """
    ...  # TODO


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
    # TODO: Run experiments
    pass

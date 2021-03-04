"""
This module contains methods to correct p-values in order to avoid the
[Multiple comparisons problem](https://en.wikipedia.org/wiki/Multiple_comparisons_problem). The code is based on
[this codebase](https://github.com/rtmdrr/replicability-analysis-NLP) corresponding to the
[Dror et al. (2017)](https://arxiv.org/abs/1709.09500) publication.
"""

# STD
from typing import List

# EXT
import numpy as np
from scipy import stats

# TODO: Add type array conversions here?


def correct_p_values(p_values: List[float], method: str = "bonferroni") -> List[float]:
    """
    Correct p-values based on Bonferroni's or Fisher's method. Bonferroni's method is most appropriate when data sets
    that the p-values originated from are dependent, and Fisher's when they are independent.

    Parameters
    ----------
    p_values: List[float]
        p-values to be corrected.
    method: str
        Method used for correction. Has to be either "bonferroni" or "fisher".

    Returns
    -------
    List[float]
        Corrected p-values.
    """
    assert method in ("bonferroni", "fisher")

    N = len(p_values)
    sorted_p_values = sorted(p_values, reverse=True)
    corrected_p_values = []

    for u in range(N):
        corrected_p_values[u] = max(
            calculate_partial_conjunction(sorted_p_values, u + 1, method),
            corrected_p_values[u - 1] if u > 0 else 0,
        )

    return corrected_p_values


def calculate_partial_conjunction(
    sorted_p_values: List[float], u: int, method: str
) -> float:
    """
    Calculate the partial conjunction p-value for u out of N.

    Parameters
    ----------
    sorted_p_values: List[float]
        Sorted p-values.
    u: int
        Number of null hypothesis.
    method: str
        Method used for correction. Has to be either "bonferroni" or "fisher".

    Returns
    -------
    float
        p-value for the partial conjunction hypothesis for u out of N.
    """
    N = len(sorted_p_values)
    p_value_selection = sorted_p_values[: (N - u + 1)]
    p_partial_u = 0

    if method == "bonferroni":
        p_partial_u = (N - u + 1) * min(p_value_selection)

    elif method == "fisher":
        p_partial_u = 1 - stats.chi2.cdf(
            -2 * np.sum(np.log(p_value_selection)), 2 * (N - u + 1)
        )

    return p_partial_u

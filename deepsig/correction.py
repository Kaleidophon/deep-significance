"""
This module contains methods to correct p-values in order to avoid the
`Multiple comparisons problem <https://en.wikipedia.org/wiki/Multiple_comparisons_problem>`_. The code is based on
`this codebase <https://github.com/rtmdrr/replicability-analysis-NLP>`_ corresponding to the
`Dror et al. (2017) <https://arxiv.org/abs/1709.09500>`_ publication.
"""

# EXT
import numpy as np

# PKD
from deepsig.conversion import score_conversion, ArrayLike

# CONST
CORRECTION_METHODS = ("holm", "bonferroni")


@score_conversion
def correct_p_values(p_values: ArrayLike, method: str = "holm") -> np.ndarray:
    """
    Correct p-values for multiple comparisons. By default, the Holm-Bonferroni method is used, which controls the
    family-wise error rate just like the Bonferroni correction, but is uniformly more powerful.

    Parameters
    ----------
    p_values: ArrayLike
        p-values to be corrected.
    method: str
        Correction method, either "holm" (default) or "bonferroni".

    Returns
    -------
    np.ndarray
        Corrected p-values, in the same order as the input.
    """
    assert (
        method in CORRECTION_METHODS
    ), "Correction method has to be one of {}, '{}' found.".format(
        ", ".join(CORRECTION_METHODS), method
    )

    if method == "holm":
        return holm_correction(p_values)

    return bonferroni_correction(p_values)


@score_conversion
def bonferroni_correction(p_values: ArrayLike) -> np.ndarray:
    """
    Correct for multiple comparisons based on Bonferroni's method, i.e. by multiplying every p-value by the number of
    comparisons.

    Parameters
    ----------
    p_values: ArrayLike
        p-values to be corrected.

    Returns
    -------
    np.ndarray
        Corrected p-values, in the same order as the input.
    """
    _check_p_values(p_values)

    N = len(p_values)
    corrected_p_values = p_values * N

    # Make sure p-values never get corrected above 1
    corrected_p_values = np.minimum(corrected_p_values, 1)

    return corrected_p_values


@score_conversion
def holm_correction(p_values: ArrayLike) -> np.ndarray:
    """
    Correct for multiple comparisons based on the Holm-Bonferroni method
    `(Holm, 1979) <https://www.jstor.org/stable/4615733>`_. The p-values are sorted ascendingly, and the u-th smallest
    p-value is multiplied by (N - u + 1). Afterwards, the corrected p-values are made monotonic, so that a smaller
    p-value never receives a larger corrected p-value than a bigger one.

    Parameters
    ----------
    p_values: ArrayLike
        p-values to be corrected.

    Returns
    -------
    np.ndarray
        Corrected p-values, in the same order as the input.
    """
    _check_p_values(p_values)

    N = len(p_values)
    p_values = p_values.copy()

    if N == 1:
        return p_values

    # In order to maintain order of p-values, remember which one was which after sorting
    sorted_indices = np.argsort(p_values, kind="stable")
    sorted_p_values = p_values[sorted_indices]
    corrected_p_values = np.ones(N)

    for u in range(N):
        corrected_p_values[u] = calculate_partial_conjunction(sorted_p_values, u + 1)

    # Make the corrected p-values monotonic: A hypothesis can only be rejected if all hypotheses with smaller p-values
    # are rejected as well
    corrected_p_values = np.maximum.accumulate(corrected_p_values)

    # Make sure p-values never get corrected above 1
    corrected_p_values = np.minimum(corrected_p_values, 1)

    # "Unsort" p-values again. This makes it easier to check which null hypotheses can be rejected.
    unsorted_p_values = np.empty(N)
    unsorted_p_values[sorted_indices] = corrected_p_values

    return unsorted_p_values


def calculate_partial_conjunction(sorted_p_values: np.ndarray, u: int) -> float:
    """
    Calculate the partial conjunction p-value for u out of N.

    Parameters
    ----------
    sorted_p_values: np.ndarray
        Sorted p-values.
    u: int
        Number of null hypothesis.

    Returns
    -------
    float
        p-value for the partial conjunction hypothesis for u out of N.
    """
    N = len(sorted_p_values)
    p_partial_u = (N - u + 1) * sorted_p_values[u - 1]

    return p_partial_u


def _check_p_values(p_values: np.ndarray) -> None:
    """
    Make sure that p-values are non-empty and lie within [0, 1].

    Parameters
    ----------
    p_values: np.ndarray
        p-values to be checked.
    """
    assert len(p_values) > 0, "List of p-values must not be empty."
    assert (0 <= p_values).all() and (
        p_values <= 1
    ).all(), "Input contains invalid p-values."

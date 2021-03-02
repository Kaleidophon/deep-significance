"""
Implementation of paired bootstrap test
[(Efron & Tibshirani, 1994)](https://cds.cern.ch/record/526679/files/0412042312_TOC.pdf).
"""

# STD
from typing import List

# EXT
import numpy as np


def bootstrap_test(
    scores_a: List[float], scores_b: List[float], num_samples: int
) -> float:
    """
    Implementation of paired bootstrap test. A p-value is being estimated by comparing the mean of scores
    for two algorithms to the means of resampled populations, where `num_samples` determines the number of
    times we resample.

    Parameters
    ----------
    scores_a: List[float]
        Scores of algorithm A.
    scores_b: List[float]
        Scores of algorithm B.
    num_samples: int
        Number of bootstrap samples used for estimation.

    Returns
    -------
    float
        Estimated p-value.
    """
    assert len(scores_a) == len(scores_b), "Scores have to be of same length."

    N = len(scores_a)
    delta = np.mean(scores_a) - np.mean(scores_b)
    num_larger = 0

    for _ in range(num_samples):
        resampled_scores_a = np.random.choice(scores_a, N)
        resampled_scores_b = np.random.choice(scores_b, N)

        new_delta = np.mean(resampled_scores_a) - np.mean(resampled_scores_b)

        if new_delta > 2 * delta:
            num_larger += 1

    p_value = num_larger / num_samples

    return p_value

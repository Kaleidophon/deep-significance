"""
Implementation of paired sign test.
"""

# STD
from typing import List

# EXT
import numpy as np


def sign_test(scores_a: List[float], scores_b: List[float], num_samples: int) -> float:
    """
    Implementation of a paired bootstrap sign test.

    Parameters
    ----------
    scores_a: List[float]
        Scores of algorithm A.
    scores_b: List[float]
        Scores of algorithm B.
    num_samples: int
        Number of permutations used for estimation.

    Returns
    -------
    float
        Estimated p-value.
    """
    assert len(scores_a) == len(scores_b), "Scores have to be of same length."
    N = len(scores_a)
    scores_a, scores_b = np.array(scores_a), np.array(scores_b)
    num_larger = 0

    # Do the permutations
    for _ in range(num_samples):
        normed_resampled_a = np.mean(np.random.choice(scores_a, N))
        normed_resampled_b = np.mean(np.random.choice(scores_b, N))

        if normed_resampled_a > normed_resampled_b:
            num_larger += 1

    p_value = (num_larger + 1) / (num_samples + 1)

    return p_value

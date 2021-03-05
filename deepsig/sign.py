"""
Implementation of paired sign test.
"""

# STD
from typing import List

# EXT
import numpy as np

# PKG
from deepsig.conversion import ArrayLike, score_conversion


@score_conversion
def sign_test(scores_a: ArrayLike, scores_b: ArrayLike, num_samples: int) -> float:
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
    num_larger = 0

    # Do the permutations
    for _ in range(num_samples):
        resampled_a = np.random.choice(scores_a, N)
        resampled_b = np.random.choice(scores_b, N)

        # A wins over B in more than 50 % of cases
        if (resampled_a > resampled_b).astype(int) > N / 2:
            num_larger += 1

    p_value = (num_larger + 1) / (num_samples + 1)

    return p_value

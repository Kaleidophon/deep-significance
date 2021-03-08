"""
Implementation of paired sign test.
"""

# EXT
import numpy as np

# PKG
from deepsig.conversion import ArrayLike, score_conversion


@score_conversion
def permutation(scores_a: ArrayLike, scores_b: ArrayLike, num_samples: int) -> float:
    """
    Implementation of a permutation-randomization test. Scores of A and B will be randomly swapped and the difference
    in samples is then compared to the original differece.

    Parameters
    ----------
    scores_a: ArrayLike
        Scores of algorithm A.
    scores_b: ArrayLike
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
    delta = np.mean(scores_a - scores_b)
    num_larger = 0

    # Do the permutations
    for _ in range(num_samples):
        # Swap entries of a and b with 50 % probability
        swapped_a, swapped_b = zip(
            *[
                (scores_a[i], scores_b[i])
                if np.random.rand() > 0.5
                else (scores_b[i], scores_a[i])
                for i in range(N)
            ]
        )

        if np.mean(swapped_a - swapped_b) <= delta:
            num_larger += 1

    p_value = (num_larger + 1) / (num_samples + 1)

    return p_value

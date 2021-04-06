"""
Implementation of paired sign test.
"""

# EXT
import numpy as np

# PKG
from deepsig.conversion import ArrayLike, score_conversion


@score_conversion
def permutation_test(
    scores_a: ArrayLike, scores_b: ArrayLike, num_samples: int = 1000
) -> float:
    """
    Implementation of a permutation-randomization test. Scores of A and B will be randomly swapped and the difference
    in samples is then compared to the original difference.

    The test is single-tailed, where we want to verify that the algorithm corresponding to `scores_a` is better than
    the one `scores_b` originated from.

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
    assert (
        len(scores_a) > 0 and len(scores_b) > 0
    ), "Both lists of scores must be non-empty."
    assert num_samples > 0, "num_samples must be positive, {} found.".format(
        num_samples
    )

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
        swapped_a, swapped_b = np.array(swapped_a), np.array(swapped_b)

        if np.mean(swapped_a - swapped_b) >= delta:
            num_larger += 1

    p_value = (num_larger + 1) / (num_samples + 1)

    return p_value

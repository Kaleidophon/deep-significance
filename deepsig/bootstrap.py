"""
Implementation of paired bootstrap test
`(Efron & Tibshirani, 1994) <https://cds.cern.ch/record/526679/files/0412042312_TOC.pdf>`_.
"""

# EXT
import numpy as np

# PKG
from deepsig.conversion import ArrayLike, score_conversion


@score_conversion
def bootstrap_test(
    scores_a: ArrayLike, scores_b: ArrayLike, num_samples: int = 1000
) -> float:
    """
    Implementation of paired bootstrap test. A p-value is being estimated by comparing the mean of scores
    for two algorithms to the means of resampled populations, where `num_samples` determines the number of
    times we resample.

    The test is single-tailed, where we want to verify that the algorithm corresponding to `scores_a` is better than
    the one `scores_b` originated from.

    Parameters
    ----------
    scores_a: ArrayLike
        Scores of algorithm A.
    scores_b: ArrrayLike
        Scores of algorithm B.
    num_samples: int
        Number of bootstrap samples used for estimation.

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
    delta = np.mean(scores_a) - np.mean(scores_b)
    num_larger = 0

    for _ in range(num_samples):
        resampled_scores_a = np.random.choice(scores_a, N)
        resampled_scores_b = np.random.choice(scores_b, N)

        new_delta = np.mean(resampled_scores_a - resampled_scores_b)

        if new_delta >= 2 * delta:
            num_larger += 1

    p_value = num_larger / num_samples

    return p_value

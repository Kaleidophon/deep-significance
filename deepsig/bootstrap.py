"""
Implementation of paired bootstrap test
`(Efron & Tibshirani, 1994) <https://cds.cern.ch/record/526679/files/0412042312_TOC.pdf>`_.
"""

# EXT
from joblib import Parallel, delayed
import numpy as np

# PKG
from deepsig.conversion import ArrayLike, score_pair_conversion

# TODO: Add seeding


@score_pair_conversion
def bootstrap_test(
    scores_a: ArrayLike, scores_b: ArrayLike, num_samples: int = 1000, num_jobs: int = 1
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
    num_jobs: int
        Number of threads that bootstrap iterations are divided among.

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

    def _bootstrap_iter(delta: float):
        """
        One bootstrap iteration. Wrapped in a function so it can be handed to joblib.Parallel.
        """
        # When running multiple jobs, modules have to be re-imported for some reason to avoid an error
        # Use dir() to check whether module is available in local scope:
        # https://stackoverflow.com/questions/30483246/how-to-check-if-a-module-has-been-imported
        if "numpy" not in dir():
            import numpy as np

        resampled_scores_a = np.random.choice(scores_a, N)
        resampled_scores_b = np.random.choice(scores_b, N)

        new_delta = np.mean(resampled_scores_a - resampled_scores_b)

        return int(new_delta >= 2 * delta)

    # Initialize worker pool and start iterations
    parallel = Parallel(n_jobs=num_jobs)
    samples = parallel(delayed(_bootstrap_iter)(delta) for _ in range(num_samples))

    p_value = sum(samples) / num_samples

    return p_value

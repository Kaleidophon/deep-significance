"""
Implementation of paired sign test.
"""

# EXT
from joblib import Parallel, delayed
import numpy as np

# PKG
from deepsig.conversion import ArrayLike, score_pair_conversion


# TODO: Add seeding


@score_pair_conversion
def permutation_test(
    scores_a: ArrayLike, scores_b: ArrayLike, num_samples: int = 1000, num_jobs: int = 1
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
    delta = np.mean(scores_a - scores_b)

    def _bootstrap_iter(delta: float):
        """
        One bootstrap iteration. Wrapped in a function so it can be handed to joblib.Parallel.
        """
        # When running multiple jobs, modules have to be re-imported for some reason to avoid an error
        # Use dir() to check whether module is available in local scope:
        # https://stackoverflow.com/questions/30483246/how-to-check-if-a-module-has-been-imported
        if "numpy" not in dir():
            import numpy as np

        swapped_a, swapped_b = zip(
            *[
                (scores_a[i], scores_b[i])
                if np.random.rand() > 0.5
                else (scores_b[i], scores_a[i])
                for i in range(N)
            ]
        )
        swapped_a, swapped_b = np.array(swapped_a), np.array(swapped_b)

        return int(np.mean(swapped_a - swapped_b) >= delta)

    # Initialize worker pool and start iterations
    parallel = Parallel(n_jobs=num_jobs)
    samples = parallel(delayed(_bootstrap_iter)(delta) for _ in range(num_samples))

    p_value = (sum(samples) + 1) / (num_samples + 1)

    return p_value

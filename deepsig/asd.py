"""
Re-implementation of Almost Stochastic Dominance (ASD) by [Dror et al. (2019)](https://arxiv.org/pdf/2010.03039.pdf).
The code here heavily borrows from their [original code base](https://github.com/rtmdrr/DeepComparison).
"""

# STD
from typing import List, Tuple


def asd(
    scores_a: List[float], scores_b: List[float], confidence_level: float = 0.05
) -> Tuple[float, float]:
    """
    Performs the Almost Stochastic Dominance test by Dror et al. (2019). The function takes two list of scores as input
    (they do not have to be of the same length) and returns the test statistic e_W1 as well as the minimum epsilon
    threshold. If e_W1 is larger or equal than the epsilon threshold, the null hypothesis is rejected (and the model
    scores_a belongs to is deemed better than the model of scores_b).

    The epsilon threshold directly depends on the number of supplied scores; thus, the more scores are used, the more
    safely we can reject the null hypothesis.

    Parameters
    ----------
    scores_a: List[float]
        Scores of algorithm A.
    scores_b: List[float]
        Scores of algorithm B.
    confidence_level: float
        Desired confidence level of test. Set to 0.05 by default.

    Returns
    -------
    Tuple[float, float]
        Return the test statistic e_W1 and the minimum epsilon threshold which is has to surpass in order for the null
        hypothesis to be rejected.
    """
    ...  # TODO


def get_epsilon_threshold(
    scores_a: List[float], scores_b: List[float], confidence_level: float
) -> float:
    """
    Get the minimum epsilon threshold which the test statistic has to surpass in order to reject the null hypothesis

    Parameters
    ----------
    scores_a: List[float]
        Scores of algorithm A.
    scores_b: List[float]
        Scores of algorithm B.
    confidence_level: float
        Desired confidence level of test.

    Returns
    -------
    float
        Minimum epsilon threshold which has to be surpassed.
    """
    ...  # TODO

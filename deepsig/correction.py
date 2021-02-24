"""
This module contains methods to correct p-values in order to avoid the
[Multiple comparisons problem](https://en.wikipedia.org/wiki/Multiple_comparisons_problem). The code is based on
[this codebase](https://github.com/rtmdrr/replicability-analysis-NLP) corresponding to the
[Dror et al. (2017)](https://arxiv.org/abs/1709.09500) publication.
"""

# STD
from typing import List


def correct_p_values(p_values: List[float], method: str = "bonferroni") -> List[float]:
    """
    Correct p-values based on Bonferroni's or Fisher's method. Bonferroni's method is most appropriate when data sets
    that the p-values originated from are dependent, and Fisher's when they are independent.

    Parameters
    ----------
    p_values: List[float]
        p-values to be corrected.
    method: str
        Method used for correction. Has to be either "bonferroni" or "fisher".

    Returns
    -------
    List[float]
        Corrected p-values.
    """
    assert method in ("bonferroni", "fisher")
    ...  # TODO

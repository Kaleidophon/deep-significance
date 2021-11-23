"""
Implement functions to help determine the right sample size for experiments.
"""

# STD
from math import sqrt


def aso_uncertainty_reduction(m_old: int, n_old: int, m_new: int, n_new: int) -> float:
    """
    Compute the reduction of uncertainty of tightness of estimate for violation ratio e_W2(F, G).
    This is based on the CLT in `del Barrio et al. (2018) <https://arxiv.org/pdf/1705.01788.pdf>`_ Theorem 2.4 / eq. 9.

    Parameters
    ----------
    m_old: int
        Old number of scores for algorithm A.
    n_old: int
        Old number of scores for algorithm B.
    m_new: int
        New number of scores for algorithm A.
    n_new: int
        New number of scores for algorithm B.

    Returns
    -------
    float
        Reduction of uncertainty / increase of tightness of estimate for violation ratio e_W2(F, G).

    """
    return sqrt((m_old + n_old) * m_new * n_new / (m_old * n_old * (m_new + n_new)))

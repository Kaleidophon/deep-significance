"""
Tests for deepsig.aso.py.
"""

# STD
import unittest

# PKG
from deepsig.aso import aso, compute_violation_ratio, get_quantile_function


class ASOTechnicalTests(unittest.TestCase):
    """
    Check technical aspects of ASO: Is the quantile function and the computation of the violation ratio
    is correct.
    """

    def setUp(self) -> None:
        ...  # TODO

    def test_compute_violation_ratio(self):
        """
        Test whether violation ratio is being computed correctly.
        """
        ...  # TODO

    def test_get_quantile_function(self):
        """
        Test whether quantile function is working correctly.
        """
        ...  # TODO


class ASOSanityTests(unittest.TestCase):
    """
    Sanity checks to test whether the ASO test behaves as expected.
    """

    def setUp(self) -> None:
        ...  # TODO

    def test_dependency_on_alpha(self):
        """
        Make sure that the minimum epsilon threshold decreases as we increase the confidence level.
        """
        ...  # TODO

    def test_dependency_on_samples(self):
        """
        Make sure that minimum epsilon threshold increases as we obtain more samples.
        """
        ...  # TODO

    def test_extreme_cases(self):
        """
        Make sure the violation rate is sensible for extreme cases (same distribution and total stochastic order).
        """
        ...  # TODO

    def test_rejection_rates(self):
        """
        Test some rejection rates based on table 1 of [del Barrio et al. (2018)](https://arxiv.org/pdf/1705.01788.pdf)
        between a Gaussian with zero mean and unit variance and another normal with variable parameters, as well as
        different minimum epsilon threshold values and sample sizes.
        """
        ...  # TODO

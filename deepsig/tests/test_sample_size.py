"""
Tests for deepsig.sample_size.py.
"""

# STD
import unittest

# PKD
from deepsig.sample_size import aso_uncertainty_reduction, bootstrap_power_analysis


class ASOUncertaintyReductionTests(unittest.TestCase):
    """
    Test computing the reduction in uncertainty around the correct estimate of the violation ratio.
    """

    def test_assertions(self):
        """
        Make sure that weird input arguments raise errors.
        """
        ...  # TODO: Implement

    def test_monotonicity(self):
        """
        Make sure that a) Increasing sample sizes always tightens the bound of the estimate and that b) the tightness of
        the bound is a monotonically increasing function of the sample sizes.
        """
        ...  # TODO: Implement


class BootstrapPowerAnalysisTests(unittest.TestCase):
    """
    Test using bootstrap power analysis to determine the right sample size.
    """

    def test_assertions(self):
        """
        Make sure that weird input arguments raise errors.
        """
        ...  # TODO: Implement

    def test_bootstrap_power_analysis(self):
        """
        Test bootstrap power analysis using different samples: Samples with very high variance should not cross the
        a certain threshold of significant comparisons. Decreasing the variance in samples should lower the percentage
        of significant comparisons.
        """
        ...  # TODO: Implement

"""
Tests for deepsig.aso.py.
"""

# STD
import unittest

# EXT
import numpy as np
from scipy.stats import wasserstein_distance, pearsonr

# PKG
from deepsig.aso import aso, compute_violation_ratio, get_quantile_function


class ASOTechnicalTests(unittest.TestCase):
    """
    Check technical aspects of ASO: Is the quantile function and the computation of the violation ratio
    is correct.
    """

    def setUp(self) -> None:
        self.num_samples = 100000

    def test_compute_violation_ratio(self):
        """
        Test whether violation ratio is being computed correctly.
        """
        samples_normal2 = np.random.normal(
            scale=2, size=self.num_samples
        )  # Scores for algorithm B
        violation_ratios = []
        inv_sqw_dists = []

        # Shift the distribution of A away (algorithm A becomes better and better)
        for loc in np.arange(0, 1, 0.05):
            samples_normal1 = np.random.normal(loc=loc, size=self.num_samples)
            violation_ratio = compute_violation_ratio(
                samples_normal1, samples_normal2, dt=0.05
            )
            w_dist = wasserstein_distance(samples_normal1, samples_normal2)
            violation_ratios.append(violation_ratio)
            inv_sqw_dists.append(1 / w_dist ** 2)

        # I didn't find a closed-form solution for the violation ratio for two gaussians - so instead I am checking
        # whether it is positively correlated with the inverse squared 1-Wasserstein distance computed via scipy
        rho, _ = pearsonr(violation_ratios, inv_sqw_dists)
        self.assertGreaterEqual(rho, 0.9)

    def test_get_quantile_function(self):
        """
        Test whether quantile function is working correctly. Values for normal distribution taken from
        https://en.wikipedia.org/wiki/Standard_normal_table.
        """
        # Test with uniform distribution
        samples_uniform = np.random.uniform(size=self.num_samples)
        quantile_func_uniform = get_quantile_function(samples_uniform)

        for x in np.arange(0, 1, 0.1):
            self.assertAlmostEqual(x, quantile_func_uniform(x), delta=0.01)

        # Test with normal distribution
        samples_normal = np.random.normal(size=self.num_samples)
        quantile_func_normal = get_quantile_function(samples_normal)

        for prob, x in [(0.84, 1), (0.5, 0), (0.31, -0.5), (0.16, -1)]:
            self.assertAlmostEqual(x, quantile_func_normal(prob), delta=0.015)


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

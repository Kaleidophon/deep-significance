"""
Tests for deepsig.aso.py.
"""

# STD
import unittest
from collections import namedtuple

# EXT
import numpy as np
from scipy.stats import wasserstein_distance, pearsonr

# PKG
from deepsig.aso import aso, compute_violation_ratio, get_quantile_function

# TYPES
RRS = namedtuple("RRS", ["eps_min", "sample_size", "scale", "loc", "rate"])

'''
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
        self.assertGreaterEqual(rho, 0.85)

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
'''


class ASOSanityChecks(unittest.TestCase):
    """
    Sanity checks to test whether the ASO test behaves as expected.
    """

    def setUp(self) -> None:
        self.num_samples = 1000
        self.num_bootstrap_iters = 500

        # For test_rejection_rates(), taken from del Barrio et al. (2018).
        self.rejection_rates_params = [
            RRS(eps_min=0.01, sample_size=100, loc=0.139, scale=1.1, rate=0),
            RRS(eps_min=0.01, sample_size=1000, loc=0.455, scale=1.5, rate=0),
            RRS(eps_min=0.01, sample_size=5000, loc=1.395, scale=2, rate=0.77),
            RRS(eps_min=0.05, sample_size=100, loc=0.091, scale=1.1, rate=0.004),
            RRS(eps_min=0.05, sample_size=1000, loc=0.697, scale=1.5, rate=0.929),
            RRS(eps_min=0.05, sample_size=5000, loc=1.395, scale=2, rate=1),
            RRS(eps_min=0.10, sample_size=100, loc=0.091, scale=1.1, rate=0.017),
            RRS(eps_min=0.10, sample_size=1000, loc=0.341, scale=1.5, rate=0.076),
            RRS(eps_min=0.10, sample_size=5000, loc=1.395, scale=2, rate=1),
        ]

    '''
    def test_extreme_cases(self):
        """
        Make sure the violation rate is sensible for extreme cases (same distribution and total stochastic order).
        """
        # Extreme case 1: Distributions are the same, SO should be extremely violated
        samples_normal2 = np.random.normal(
            size=self.num_samples
        )  # Scores for algorithm B
        samples_normal1 = np.random.normal(
            size=self.num_samples
        )  # Scores for algorithm A

        vr, eps_min = aso(samples_normal1, samples_normal2, num_bootstrap_iterations=self.num_bootstrap_iters)
        print(vr, eps_min)
        self.assertGreaterEqual(vr, 0.9)
        self.assertAlmostEqual(eps_min, 1, delta=0.001)

        # Extreme case 2: Distribution for A is wayyy better, should basically be SO
        samples_normal3 = np.random.normal(
            loc=5, scale=0.1, size=self.num_samples
        )  # New scores for algorithm A
        vr2, eps_min2 = aso(samples_normal3, samples_normal2, num_bootstrap_iterations=self.num_bootstrap_iters)
        self.assertAlmostEqual(vr2, 0, delta=0.01)
        self.assertAlmostEqual(eps_min2, 0, delta=0.01)
    '''

    '''
    def test_dependency_on_alpha(self):
        """
        Make sure that the minimum epsilon threshold increases as we increase the confidence level.
        """
        samples_normal1 = np.random.normal(
            loc=0.1, size=self.num_samples
        )  # Scores for algorithm A
        samples_normal2 = np.random.normal(
            scale=2, size=self.num_samples
        )  # Scores for algorithm B

        violation_ratios = []
        min_epsilons = []
        for alpha in np.arange(0.8, 0.1, -0.1):
            vr, min_eps = aso(
                samples_normal1,
                samples_normal2,
                confidence_level=alpha,
                num_bootstrap_iterations=100,
            )
            violation_ratios.append(vr)
            min_epsilons.append(min_eps)

        self.assertEqual(
            len(set(violation_ratios)), 1
        )  # Check that violation ratio stays constant
        self.assertEqual(
            list(sorted(min_epsilons)), min_epsilons
        )  # Make sure min_epsilon decreases
    '''

    def test_rejection_rates(self):
        """
        Test some rejection rates based on table 1 of [del Barrio et al. (2018)](https://arxiv.org/pdf/1705.01788.pdf)
        between a Gaussian with zero mean and unit variance and another normal with variable parameters, as well as
        different minimum epsilon threshold values and sample sizes.
        """
        # TODO: This is still untested and quite slow
        num_simulations = 1000

        for rrs in self.rejection_rates_params:
            num_rejections = 0

            for _ in range(num_simulations):
                print(_)
                samples_normal1 = np.random.normal(
                    size=rrs.sample_size
                )  # Scores for algorithm A
                samples_normal2 = np.random.normal(
                    loc=rrs.loc, scale=rrs.scale, size=rrs.sample_size
                )  # Scores for algorithm B
                vr, _ = aso(
                    samples_normal1,
                    samples_normal2,
                    num_bootstrap_iterations=100,
                    num_samples_a=100,
                    num_samples_b=100,
                )
                return

                if vr > rrs.eps_min:
                    num_rejections += 1

            rejection_rate = num_rejections / num_simulations
            print(rejection_rate)
            self.assertAlmostEqual(rejection_rate, rrs.rate, delta=0.05)

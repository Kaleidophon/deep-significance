"""
Tests for deepsig.aso.py.
"""

# STD
import unittest

# EXT
import numpy as np
from scipy.stats import wasserstein_distance, pearsonr

# PKG
from deepsig.aso import (
    aso,
    compute_violation_ratio,
    get_quantile_function,
    FAST_QUANTILE_WARN_INTERVAL,
)


class ASOTechnicalTests(unittest.TestCase):
    """
    Check technical aspects of ASO: Is the quantile function and the computation of the violation ratio
    is correct.
    """

    def setUp(self) -> None:
        self.num_samples = 100000

    def test_assertions(self):
        """
        Make sure that invalid input arguments raise an error.
        """
        with self.assertRaises(AssertionError):
            aso([], [1, 3])

        with self.assertRaises(AssertionError):
            aso([3, 4], [])

        with self.assertRaises(AssertionError):
            aso([1, 2, 3], [3, 4, 5], num_samples=-1, show_progress=False)

        with self.assertRaises(AssertionError):
            aso([1, 2, 3], [3, 4, 5], num_samples=0, show_progress=False)

        with self.assertRaises(AssertionError):
            aso([1, 2, 3], [3, 4, 5], num_bootstrap_iterations=-1, show_progress=False)

        with self.assertRaises(AssertionError):
            aso([1, 2, 3], [3, 4, 5], num_bootstrap_iterations=0, show_progress=False)

        with self.assertRaises(AssertionError):
            aso([1, 2, 3], [3, 4, 5], build_quantile="foobar", show_progress=False)

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
            self.assertAlmostEqual(x, quantile_func_normal(prob), delta=0.02)

    def test_quantile_function_building(self):
        """
        Test whether building the quantile functions exactly or fast is sufficiently close enough.
        """
        # Define parameters of gaussian for which we will test if the two methods are sufficiently close.
        parameters = [
            ((5, 0.1), (0, 1)),
            ((0, 0.5), (0, 1)),
            ((2, 2), (1, 1)),
            ((-0.5, 0.1), (-0.6, 0.2)),
            ((0.5, 0.21), (0.7, 0.1)),
            ((0.1, 0.3), (0.2, 0.1)),
        ]

        for (loc1, scale1), (loc2, scale2) in parameters:
            samples_normal1 = np.random.normal(
                loc=loc1, scale=scale1, size=10
            )  # New scores for algorithm A
            samples_normal2 = np.random.normal(
                loc=loc2, scale=scale2, size=10
            )  # Scores for algorithm B

            eps_min1 = aso(
                samples_normal1,
                samples_normal2,
                num_bootstrap_iterations=500,
                build_quantile="exact",
                show_progress=False,
            )
            eps_min2 = aso(
                samples_normal1,
                samples_normal2,
                num_bootstrap_iterations=500,
                build_quantile="fast",
                show_progress=False,
            )
            self.assertAlmostEqual(
                eps_min1, eps_min2, delta=FAST_QUANTILE_WARN_INTERVAL
            )


class ASOSanityChecks(unittest.TestCase):
    """
    Sanity checks to test whether the ASO test behaves as expected.
    """

    def setUp(self) -> None:
        self.num_samples = 1000
        self.num_bootstrap_iters = 500

    def test_extreme_cases(self):
        """
        Make sure the violation rate is sensible for extreme cases (same distribution and total stochastic order).
        """
        # Extreme case 1: Distributions are basically the same, SO should be extremely violated
        samples_normal2 = np.random.normal(
            size=self.num_samples
        )  # Scores for algorithm B
        samples_normal1 = np.random.normal(
            size=self.num_samples
        )  # Scores for algorithm A

        eps_min = aso(
            samples_normal1,
            samples_normal1 + 1e-8,
            num_bootstrap_iterations=self.num_bootstrap_iters,
            show_progress=False,
        )
        self.assertAlmostEqual(eps_min, 1, delta=0.001)

        # Extreme case 2: Distribution for A is wayyy better, should basically be SO
        samples_normal3 = np.random.normal(
            loc=5, scale=0.1, size=self.num_samples
        )  # New scores for algorithm A
        eps_min2 = aso(
            samples_normal3,
            samples_normal2,
            num_bootstrap_iterations=self.num_bootstrap_iters,
            show_progress=False,
        )
        self.assertAlmostEqual(eps_min2, 0, delta=0.01)

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

        min_epsilons = []
        for alpha in np.arange(0.8, 0.1, -0.1):
            min_eps = aso(
                samples_normal1,
                samples_normal2,
                confidence_level=alpha,
                num_bootstrap_iterations=100,
                show_progress=False,
            )
            min_epsilons.append(min_eps)

        self.assertEqual(
            list(sorted(min_epsilons)), min_epsilons
        )  # Make sure min_epsilon decreases

    def test_symmetry(self):
        """
        Test whether ASO(A, B, alpha) = 1 - ASO(B, A, alpha) holds.
        """
        parameters = [
            ((5, 0.1), (0, 1)),
            ((0, 0.5), (0, 1)),
            ((2, 2), (1, 1)),
            ((-0.5, 0.1), (-0.6, 0.2)),
            ((0.5, 0.21), (0.7, 0.1)),
            ((0.1, 0.3), (0.2, 0.1)),
        ]

        for (loc1, scale1), (loc2, scale2) in parameters:
            samples_normal1 = np.random.normal(
                loc=loc1, scale=scale1, size=50
            )  # New scores for algorithm A
            samples_normal2 = np.random.normal(
                loc=loc2, scale=scale2, size=50
            )  # Scores for algorithm B

            eps_min1 = aso(
                samples_normal1,
                samples_normal2,
                build_quantile="fast",
                show_progress=False,
            )
            eps_min2 = aso(
                samples_normal2,
                samples_normal1,
                build_quantile="fast",
                show_progress=False,
            )
            self.assertAlmostEqual(eps_min1, 1 - eps_min2, delta=0.01)

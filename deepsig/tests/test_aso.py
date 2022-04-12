"""
Tests for deepsig.aso.py.
"""

# STD
from itertools import product
import unittest

# EXT
import numpy as np
import torch
import tensorflow as tf

# import jax.numpy as jnp
from scipy.stats import wasserstein_distance, pearsonr, norm, laplace, rayleigh

# PKG
from deepsig.aso import (
    aso,
    multi_aso,
    compute_violation_ratio,
    get_quantile_function,
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
            aso([1, 2, 3], [3, 4, 5], num_bootstrap_iterations=-1, show_progress=False)

        with self.assertRaises(AssertionError):
            aso([1, 2, 3], [3, 4, 5], num_bootstrap_iterations=0, show_progress=False)

        with self.assertRaises(AssertionError):
            aso([1, 2, 3], [3, 4, 5], num_jobs=0, show_progress=False)

    def test_argument_combos(self):
        """
        Try different combinations of inputs arguments for compute_violation_ratio().
        """
        scores_a = np.random.normal(size=5)
        scores_b = np.random.normal(size=5)
        quantile_func_a = norm.ppf
        quantile_func_b = norm.ppf

        # All of these should work
        for kwarg1, kwarg2 in product(
            [{"scores_a": scores_a}, {"quantile_func_a": quantile_func_a}],
            [{"scores_b": scores_b}, {"quantile_func_b": quantile_func_b}],
        ):
            compute_violation_ratio(**{**kwarg1, **kwarg2})

        # These should create errors
        with self.assertRaises(AssertionError):
            compute_violation_ratio(scores_a=scores_a, quantile_func_a=quantile_func_a)

        with self.assertRaises(AssertionError):
            compute_violation_ratio(scores_b=scores_b, quantile_func_b=quantile_func_b)

    def test_compute_violation_ratio_correlation(self):
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

    def test_compute_violation_ratio_exact(self):
        """
        Test the value of the violation ratio given some exact CDFs.
        """
        test_dists = [
            (
                np.random.normal,
                norm.ppf,
                {"loc": 0.275, "scale": 1.5},
                {"loc": 0.25, "scale": 1},
            ),
            (
                np.random.laplace,
                laplace.ppf,
                {"loc": 0.275, "scale": 1.5},
                {"loc": 0.25, "scale": 1},
            ),
            (np.random.rayleigh, rayleigh.ppf, {"scale": 1.05}, {"scale": 1}),
        ]

        for sample_func, ppf, params_a, params_b in test_dists:
            quantile_func_a = lambda x: ppf(x, **params_a)
            quantile_func_b = lambda x: ppf(x, **params_b)
            violation_ratio_ab_exact = compute_violation_ratio(
                quantile_func_a=quantile_func_a, quantile_func_b=quantile_func_b
            )
            violation_ratio_ba_exact = compute_violation_ratio(
                quantile_func_a=quantile_func_b, quantile_func_b=quantile_func_a
            )

            samples_a = sample_func(size=self.num_samples, **params_a)
            samples_b = sample_func(size=self.num_samples, **params_b)
            violation_ratio_ab_sampled = compute_violation_ratio(
                scores_a=samples_a, scores_b=samples_b
            )
            violation_ratio_ba_sampled = compute_violation_ratio(
                scores_a=samples_b, scores_b=samples_a
            )

            # Check symmetries
            self.assertAlmostEqual(
                violation_ratio_ab_exact, 1 - violation_ratio_ba_exact, delta=0.05
            )
            self.assertAlmostEqual(
                violation_ratio_ab_sampled, 1 - violation_ratio_ba_sampled, delta=0.05
            )

            # Check closeness to exact value
            self.assertAlmostEqual(
                violation_ratio_ab_exact, violation_ratio_ab_sampled, delta=0.05
            )
            self.assertAlmostEqual(
                violation_ratio_ba_exact, violation_ratio_ba_sampled, delta=0.05
            )

    def test_get_quantile_function(self):
        """
        Test whether quantile function is working correctly. Values for normal distribution taken from
        https://en.wikipedia.org/wiki/Standard_normal_table.
        """
        # Test with uniform distribution
        samples_uniform = np.random.uniform(size=self.num_samples)
        quantile_func_uniform = get_quantile_function(samples_uniform)

        for prob in np.arange(0, 1, 0.1):  # For uniform, prob == x
            self.assertAlmostEqual(prob, quantile_func_uniform(prob), delta=0.01)

        # Test with normal distribution
        samples_normal = np.random.normal(size=self.num_samples)
        quantile_func_normal = get_quantile_function(samples_normal)

        for prob, x in [(0.84, 1), (0.5, 0), (0.31, -0.5), (0.16, -1)]:
            self.assertAlmostEqual(x, quantile_func_normal(prob), delta=0.15)

    def test_quantile_function_building(self):
        """
        Test whether building the quantile functions with one or multiple jobs returns roughly the same value.
        """
        # Define parameters of gaussian for which we will test if the two methods are sufficiently close.
        parameters = [
            ((0, 0.5), (0, 1)),
            ((-0.5, 0.1), (-0.6, 0.2)),
            ((0.5, 0.21), (0.7, 0.1)),
            ((0.1, 0.3), (0.2, 0.1)),
        ]

        for (loc1, scale1), (loc2, scale2) in parameters:
            samples_normal1 = np.random.normal(
                loc=loc1, scale=scale1, size=500
            )  # New scores for algorithm A
            samples_normal2 = np.random.normal(
                loc=loc2, scale=scale2, size=500
            )  # Scores for algorithm B

            # Set seed to make sure that any variance just comes from the difference between jobs
            seed = np.random.randint(0, 10000)

            eps_min1 = aso(
                samples_normal1,
                samples_normal2,
                num_bootstrap_iterations=500,
                num_jobs=1,
                show_progress=False,
                seed=seed,
            )

            eps_min2 = aso(
                samples_normal1,
                samples_normal2,
                num_bootstrap_iterations=500,
                num_jobs=2,
                show_progress=False,
                seed=seed,
            )
            self.assertAlmostEqual(eps_min1, eps_min2, delta=0.12)


class MultiASOTests(unittest.TestCase):
    """
    Test different aspects of multi_aso().
    """

    def setUp(self) -> None:
        self.aso_kwargs = {
            "num_bootstrap_iterations": 100,
            "num_jobs": 4,
            "show_progress": False,
        }
        self.num_models = 3
        self.num_seeds = 100
        np.random.seed(5678)
        self.scores = [
            np.random.normal(loc=0.3, scale=0.2, size=self.num_seeds).tolist()
            for _ in range(self.num_models)
        ]
        self.scores_dict = {
            "model{}".format(i): scores for i, scores in enumerate(self.scores)
        }
        self.scores_numpy = np.array(self.scores)
        self.scores_torch = torch.from_numpy(self.scores_numpy)
        self.scores_tensorflow = tf.convert_to_tensor(self.scores_numpy)
        self.all_score_types = [
            self.scores,
            self.scores_dict,
            self.scores_numpy,
            self.scores_torch,
            self.scores_tensorflow,
        ]
        # self.scores_jax = jnp.array(self.scores_numpy)

    def test_score_types(self):
        """
        Test different types for the scores argument.
        """
        for scores in self.all_score_types:
            multi_aso(scores, **self.aso_kwargs)

    def test_bonferroni_correction(self):
        """
        Test flag that toggles the use of the Bonferroni correction.
        """
        seed = 123
        corrected_scores = multi_aso(self.scores_numpy, seed=seed, **self.aso_kwargs)
        uncorrected_scores = multi_aso(
            self.scores_numpy, seed=seed, use_bonferroni=False, **self.aso_kwargs
        )
        self.assertTrue(np.all(corrected_scores >= uncorrected_scores))

    def test_result_df(self):
        """
        Test the creation of a results DataFrame.
        """
        seed = 5555
        eps_min = multi_aso(self.scores_dict, seed=seed, **self.aso_kwargs)
        eps_min_df = multi_aso(
            self.scores_dict, seed=seed, **self.aso_kwargs, return_df=True
        )

        self.assertEqual(list(self.scores_dict.keys()), list(eps_min_df.columns))
        self.assertEqual(list(self.scores_dict.keys()), list(eps_min_df.index))
        self.assertTrue(np.all(eps_min == eps_min_df.to_numpy()))


class ASOSanityChecks(unittest.TestCase):
    """
    Sanity checks to test whether the ASO test behaves as expected.
    """

    def setUp(self) -> None:
        self.num_samples = 1000
        self.num_bootstrap_iters = 1000

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
            num_jobs=4,
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
            num_jobs=4,
        )
        self.assertAlmostEqual(eps_min2, 0, delta=0.01)

    def test_dependency_on_confidence_level(self):
        """
        Make sure that the minimum epsilon threshold decreases as we increase the confidence level.
        """
        samples_normal1 = np.random.normal(
            loc=0.1, size=self.num_samples
        )  # Scores for algorithm A
        samples_normal2 = np.random.normal(
            scale=2, size=self.num_samples
        )  # Scores for algorithm B

        min_epsilons = []
        seed = 6666
        for confidence_level in np.arange(0.1, 0.8, 0.1):
            min_eps = aso(
                samples_normal1,
                samples_normal2,
                confidence_level=confidence_level,
                num_bootstrap_iterations=100,
                show_progress=False,
                num_jobs=4,
                seed=seed,
            )
            min_epsilons.append(min_eps)

        self.assertEqual(
            list(sorted(min_epsilons)), min_epsilons
        )  # Make sure min_epsilon decreases

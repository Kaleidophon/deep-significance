"""
Tests for deepsig.sample_size.py.
"""

# STD
import unittest

# EXT
import numpy as np
from scipy.stats import ks_2samp

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
        with self.assertRaises(AssertionError):
            aso_uncertainty_reduction(-1, 0, 1, 2)

        with self.assertRaises(AssertionError):
            aso_uncertainty_reduction(1.1, 2.3, 4.5, 4.1)

    def test_monotonicity(self):
        """
        Make sure that a) Increasing sample sizes always tightens the bound of the estimate and that b) the tightness of
        the bound is a monotonically increasing function of the sample sizes.
        """
        base_samples = [1, 1, 1, 1]

        for var in [2, 3]:
            reductions = []

            for increase in [2, 4, 20, 50, 200]:
                samples = list(base_samples)
                samples[var] += increase  # Increase sample size

                reductions.append(aso_uncertainty_reduction(*samples))

            # Check that increase in tightness is monotonically increasing with sample size
            self.assertEqual(reductions, list(sorted(reductions)))


class BootstrapPowerAnalysisTests(unittest.TestCase):
    """
    Test using bootstrap power analysis to determine the right sample size.
    """

    def test_assertions(self):
        """
        Make sure that weird input arguments raise errors.
        """
        with self.assertRaises(AssertionError):
            bootstrap_power_analysis([])

        samples = np.random.randn(3)
        with self.assertRaises(AssertionError):
            bootstrap_power_analysis(samples, num_bootstrap_iterations=0)

        with self.assertRaises(AssertionError):
            bootstrap_power_analysis(samples, scalar=0.8)

    def test_bootstrap_power_analysis(self):
        """
        Test bootstrap power analysis using different samples: Samples with very high variance should not cross the
        a certain threshold of significant comparisons. Decreasing the variance in samples should lower the percentage
        of significant comparisons.
        """
        seed = 1234
        np.random.seed(seed)

        # Test bad sample with high variance
        bad_sample = np.random.normal(0, 20, 5)
        power = bootstrap_power_analysis(bad_sample, show_progress=False, seed=seed)
        self.assertLessEqual(power, 0.2)

        # Test good sample with low variance
        good_sample = np.random.normal(0, 0.1, 50)
        power2 = bootstrap_power_analysis(good_sample, show_progress=False, seed=seed)
        self.assertGreater(power2, power)  # Power should be higher

        # Test with different significance threshold
        power3 = bootstrap_power_analysis(
            bad_sample, show_progress=False, seed=seed, significance_threshold=1
        )
        self.assertEqual(power3, 1)

        power4 = bootstrap_power_analysis(
            good_sample, show_progress=False, seed=seed, significance_threshold=0
        )
        self.assertEqual(power4, 0)

        # Test with different significance test
        # Only fails when it throws an exception
        bootstrap_power_analysis(
            good_sample,
            show_progress=False,
            seed=seed,
            significance_test=lambda scores_a, scores_b: ks_2samp(
                data1=scores_a, data2=scores_b
            )[1],
        )

        # Test with different scalar for lifting
        power5 = bootstrap_power_analysis(
            bad_sample, show_progress=False, seed=seed, scalar=1.1
        )
        self.assertLessEqual(power5, power)

        power6 = bootstrap_power_analysis(
            bad_sample, show_progress=False, seed=seed, scalar=1.5
        )
        self.assertGreaterEqual(power6, power5)

        # Test monotonicity - power should increase as a function of sample size
        powers = []
        for sample_size in [5, 10, 20, 50, 100, 200, 500]:
            samples = np.random.normal(0, 5, sample_size)
            powers.append(
                bootstrap_power_analysis(samples, show_progress=False, seed=seed)
            )

        self.assertEqual(powers, list(sorted(powers)))

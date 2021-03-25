"""
Tests for deepsig.bootstrap.py and deepsig.permutation.py.
"""

# STD
import unittest

# EXT
import numpy as np

# PKG
from deepsig import bootstrap_test, permutation_test


class BootstrapPermutationTests(unittest.TestCase):
    """
    Test the bootstrap permutation-randomization test.
    """

    def setUp(self) -> None:
        self.num_samples = 100

    def test_assertions(self):
        """
        Make sure that invalid input arguments raise an error.
        """
        # Assertions to test here: Score arrays must be non-empty and of same length
        test_arrays = [([], [3, 4]), ([1, 3], []), ([], []), ([3, 4, 5], [1, 2])]

        for scores_a, scores_b in test_arrays:
            with self.assertRaises(AssertionError):
                bootstrap_test(scores_a, scores_b)

            with self.assertRaises(AssertionError):
                permutation_test(scores_a, scores_b)

    def test_extreme_cases(self):
        """
        Test extreme cases where distributions are the same / vastly different. Also make sure that p-value
        is lower for bootstrap test when distributions are indeed different as it is a more powerful test.
        """
        # Same uniform distributions
        samples_a1 = np.random.uniform(0, 5, self.num_samples)
        samples_b1 = np.random.uniform(0, 5, self.num_samples)

        p_value_perm1 = permutation_test(samples_a1, samples_b1)
        p_value_boot1 = bootstrap_test(samples_a1, samples_b1)

        self.assertGreater(p_value_perm1, 0.05)
        self.assertGreater(p_value_boot1, 0.05)

        # Different uniform distributions
        samples_c1 = np.random.uniform(5, 10, self.num_samples)

        p_value_perm2 = permutation_test(samples_c1, samples_a1)
        p_value_boot2 = bootstrap_test(samples_c1, samples_a1)

        self.assertLessEqual(p_value_perm2, 0.05)
        self.assertLessEqual(p_value_boot2, 0.05)

        # Compare p-values
        self.assertLessEqual(p_value_boot2, p_value_perm2)

        # Same normal distributions
        samples_a2 = np.random.normal(size=self.num_samples)
        samples_b2 = np.random.normal(size=self.num_samples)

        p_value_perm3 = permutation_test(samples_a2, samples_b2)
        p_value_boot3 = bootstrap_test(samples_a2, samples_b2)

        self.assertGreater(p_value_perm3, 0.05)
        self.assertGreater(p_value_boot3, 0.05)

        # Different uniform distributions
        samples_c2 = np.random.normal(loc=5, size=self.num_samples)

        p_value_perm4 = permutation_test(samples_c2, samples_a2)
        p_value_boot4 = bootstrap_test(samples_c2, samples_a2)

        self.assertLessEqual(p_value_perm4, 0.05)
        self.assertLessEqual(p_value_boot4, 0.05)

        # Compare p-values
        self.assertLessEqual(p_value_boot4, p_value_perm4)

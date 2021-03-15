"""
Tests for deepsig.correction.py
"""

# STD
import unittest

# EXT
import numpy as np

# PKD
from deepsig import correct_p_values


class CorrectionTests(unittest.TestCase):
    """
    Test Bonferroni and Fisher correction for p-values.
    """

    def test_assertions(self):
        """
        Make sure that invalid input arguments raise an error.
        """

        with self.assertRaises(AssertionError):
            correct_p_values([])

        with self.assertRaises(AssertionError):
            correct_p_values([0.5, 0.5], method="euler")

        with self.assertRaises(AssertionError):
            correct_p_values([-0.4, 0.5])

        with self.assertRaises(AssertionError):
            correct_p_values([0.3, 1.2])

    def test_bonferroni_correction(self):
        """
        Test whether the Bonferroni correction works as expected.
        """
        # 1. Test single p-value
        p_values1 = np.random.rand(1)
        self.assertEqual(p_values1, correct_p_values(p_values1, method="bonferroni"))

        # 2. Test identical p-values
        p_values2 = np.ones(5) * np.random.rand(1) / 5
        corrected_p_values2 = correct_p_values(p_values2, method="bonferroni")
        self.assertTrue((p_values2 * np.arange(5, 0, -1) == corrected_p_values2).all())

        # 3. Test different p-values
        p_values3 = np.random.rand(5) / 5
        p_values3.sort()  # Sort values here already so that the multiplication with np.arange for the test works
        corrected_p_values3 = correct_p_values(p_values3, method="bonferroni")
        self.assertTrue((p_values3 * np.arange(5, 0, -1) == corrected_p_values3).all())

        # Make sure absurdly high p-values don't corrected over 1
        p_values4 = np.ones(4) - 1e-4
        for p_corrected in correct_p_values(p_values4, method="bonferroni"):
            self.assertAlmostEqual(p_corrected, 1, delta=0.01)

    def test_fisher_correction(self):
        """
        Test whether the Fisher correction works as expected.
        """
        ...  # TODO: p-values need not decrease after fisher correction. Thus, try to write a new test here instead
        """
        # 1. Test single p-value
        p_values1 = np.random.rand(1)
        self.assertEqual(p_values1, correct_p_values(p_values1, method="fisher"))

        # 2. Test identical p-values - less precise, just check that values decreased
        p_values2 = np.ones(5) * np.random.rand(1) / 5
        corrected_p_values2 = correct_p_values(p_values2, method="fisher")
        self.assertTrue((p_values2 >= corrected_p_values2 - 1e-8).all())

        # 3. Test different p-values
        p_values3 = np.random.rand(5) / 5
        corrected_p_values3 = correct_p_values(p_values3, method="fisher")
        self.assertTrue((p_values3 >= corrected_p_values3).all())

        # 4. Make sure low p-values don't corrected under 0
        p_values4 = np.zeros(4) + 1e-4
        for p_corrected in correct_p_values(p_values4, method="fisher"):
            self.assertAlmostEqual(p_corrected, 0, delta=0.01)
        """

"""
Tests for deepsig.correction.py
"""

# STD
import unittest

# EXT
import numpy as np

# PKD
from deepsig import bonferroni_correction


class CorrectionTests(unittest.TestCase):
    """
    Test Bonferroni correction for p-values.
    """

    def test_assertions(self):
        """
        Make sure that invalid input arguments raise an error.
        """

        with self.assertRaises(AssertionError):
            bonferroni_correction([])

        with self.assertRaises(AssertionError):
            bonferroni_correction([-0.4, 0.5])

        with self.assertRaises(AssertionError):
            bonferroni_correction([0.3, 1.2])

    def test_bonferroni_correction(self):
        """
        Test whether the Bonferroni correction works as expected.
        """
        # 1. Test single p-value
        p_values1 = np.random.rand(1)
        self.assertEqual(p_values1, bonferroni_correction(p_values1))

        # 2. Test identical p-values
        p_values2 = np.ones(5) * np.random.rand(1) / 5
        corrected_p_values2 = bonferroni_correction(p_values2)
        self.assertTrue((p_values2 * np.arange(5, 0, -1) == corrected_p_values2).all())

        # 3. Test different p-values
        p_values3 = np.random.rand(5) / 5
        p_values3.sort()  # Sort values here already so that the multiplication with np.arange for the test works
        corrected_p_values3 = bonferroni_correction(p_values3)
        self.assertTrue((p_values3 * np.arange(5, 0, -1) == corrected_p_values3).all())

        # Make sure absurdly high p-values don't corrected over 1
        p_values4 = np.ones(4) - 1e-4
        for p_corrected in bonferroni_correction(p_values4):
            self.assertAlmostEqual(p_corrected, 1, delta=0.01)

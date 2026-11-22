"""
Tests for deepsig.correction.py
"""

# STD
import unittest

# EXT
import numpy as np

# PKD
from deepsig import bonferroni_correction, correct_p_values, holm_correction


class CorrectionTests(unittest.TestCase):
    """
    Test Bonferroni and Holm-Bonferroni correction for p-values.
    """

    def test_assertions(self):
        """
        Make sure that invalid input arguments raise an error.
        """
        for correction_func in (bonferroni_correction, holm_correction):
            with self.assertRaises(AssertionError):
                correction_func([])

            with self.assertRaises(AssertionError):
                correction_func([-0.4, 0.5])

            with self.assertRaises(AssertionError):
                correction_func([0.3, 1.2])

        with self.assertRaises(AssertionError):
            correct_p_values([0.1, 0.2], method="sidak")

    def test_bonferroni_correction(self):
        """
        Test whether the Bonferroni correction works as expected.
        """
        # 1. Test single p-value
        p_values1 = np.random.rand(1)
        self.assertEqual(p_values1, bonferroni_correction(p_values1))

        # 2. Test that all p-values are multiplied by the number of comparisons, independent of their order
        p_values2 = np.random.rand(5) / 5
        corrected_p_values2 = bonferroni_correction(p_values2)
        self.assertTrue(np.allclose(p_values2 * 5, corrected_p_values2))

        # 3. Test known values
        corrected_p_values3 = bonferroni_correction([0.01, 0.011, 0.012])
        self.assertTrue(np.allclose([0.03, 0.033, 0.036], corrected_p_values3))

        # Make sure absurdly high p-values don't corrected over 1
        p_values4 = np.ones(4) - 1e-4
        for p_corrected in bonferroni_correction(p_values4):
            self.assertAlmostEqual(p_corrected, 1, delta=0.01)

    def test_holm_correction(self):
        """
        Test whether the Holm-Bonferroni correction works as expected.
        """
        # 1. Test single p-value
        p_values1 = np.random.rand(1)
        self.assertEqual(p_values1, holm_correction(p_values1))

        # 2. Test identical p-values: Monotonicity means they all receive the largest multiplier
        p_values2 = np.ones(5) * np.random.rand(1) / 5
        corrected_p_values2 = holm_correction(p_values2)
        self.assertTrue(np.allclose(p_values2 * 5, corrected_p_values2))

        # 3. Test sorted p-values, where no monotonicity adjustment is necessary
        corrected_p_values3 = holm_correction([0.01, 0.02, 0.04])
        self.assertTrue(np.allclose([0.03, 0.04, 0.04], corrected_p_values3))

        # 4. Test that monotonicity is enforced
        corrected_p_values4 = holm_correction([0.01, 0.011, 0.012])
        self.assertTrue(np.allclose([0.03, 0.03, 0.03], corrected_p_values4))

        # 5. Test that the original order is restored for unsorted input
        # Sorted: [0.01, 0.02, 0.03] -> [0.03, 0.04, 0.03] -> monotonic: [0.03, 0.04, 0.04]
        corrected_p_values5 = holm_correction([0.03, 0.01, 0.02])
        self.assertTrue(np.allclose([0.04, 0.03, 0.04], corrected_p_values5))

        # 6. Test that a larger p-value never receives a smaller corrected p-value, and that Holm is never more
        # conservative than Bonferroni
        p_values6 = np.random.rand(10) / 10
        corrected_p_values6 = holm_correction(p_values6)
        sorted_indices = np.argsort(p_values6)
        self.assertTrue((np.diff(corrected_p_values6[sorted_indices]) >= 0).all())
        self.assertTrue((corrected_p_values6 <= bonferroni_correction(p_values6)).all())

        # Make sure absurdly high p-values don't corrected over 1
        p_values7 = np.ones(4) - 1e-4
        for p_corrected in holm_correction(p_values7):
            self.assertAlmostEqual(p_corrected, 1, delta=0.01)

    def test_correct_p_values(self):
        """
        Test that correct_p_values() uses Holm-Bonferroni by default and dispatches to the right method.
        """
        p_values = np.random.rand(5) / 5

        self.assertTrue(
            np.allclose(holm_correction(p_values), correct_p_values(p_values))
        )
        self.assertTrue(
            np.allclose(
                holm_correction(p_values), correct_p_values(p_values, method="holm")
            )
        )
        self.assertTrue(
            np.allclose(
                bonferroni_correction(p_values),
                correct_p_values(p_values, method="bonferroni"),
            )
        )

"""
Test for deepsig.conversion.py.
"""

# STD
import unittest

# EXT
import numpy as np

# PKG
from deepsig import aso, bootstrap_test, permutation_test, bonferroni_correction
from deepsig.conversion import ArrayLike


class ConversionTests(unittest.TestCase):
    """
    Test that arrays of different data types are converted correctly internally.
    """

    def setUp(self) -> None:
        self.scores_a = [0.6, 4, 5, 2.3]
        self.scores_b = [0.18, 2, 2.1, 3]
        self.p_values = [0.3, 0.71, 0.04]

    @staticmethod
    def _aso_wrapper(scores_a: ArrayLike, scores_b: ArrayLike):
        return aso(
            scores_a,
            scores_b,
            num_samples=1,
            num_bootstrap_iterations=1,
            show_progress=False,
        )

    def test_python_compatability(self):
        """
        Test compatibility of functions with native python types.
        """
        # Just run and check these call don't throw an error
        bootstrap_test(set(self.scores_a), set(self.scores_b))
        permutation_test(set(self.scores_a), set(self.scores_b))
        aso(
            set(self.scores_a),
            set(self.scores_b),
            num_samples=1,
            num_bootstrap_iterations=1,
            show_progress=False,
        )
        bonferroni_correction(set(self.p_values))

        bootstrap_test(tuple(self.scores_a), tuple(self.scores_b))
        permutation_test(tuple(self.scores_a), tuple(self.scores_b))
        aso(
            tuple(self.scores_a),
            tuple(self.scores_b),
            num_samples=1,
            num_bootstrap_iterations=1,
            show_progress=False,
        )
        bonferroni_correction(tuple(self.p_values))

    def test_numpy_array_shapes(self):
        """
        Test different numpy array shapes.
        """
        # These should work
        correct1_scores_a = np.array(self.scores_a)
        correct1_scores_b = np.array(self.scores_b)
        correct2_scores_a = correct1_scores_a[..., np.newaxis]
        correct2_scores_b = correct1_scores_b[..., np.newaxis]

        for test_func in [bootstrap_test, permutation_test, self._aso_wrapper]:
            for scores_a, scores_b in [
                (correct1_scores_a, correct1_scores_b),
                (correct2_scores_a, correct2_scores_b),
            ]:
                test_func(scores_a, scores_b)

        correct1_p_values = np.array(self.p_values)
        correct2_p_values = correct1_p_values[..., np.newaxis]

        bonferroni_correction(correct1_p_values)
        bonferroni_correction(correct2_p_values)

        # These should fail
        incorrect_scores_a = np.ones((2, 4)) * self.scores_a
        incorrect_scores_b = np.ones((2, 4)) * self.scores_b
        incorrect_p_values = np.ones((2, 3)) * self.p_values

        for test_func in [bootstrap_test, permutation_test, self._aso_wrapper]:
            with self.assertRaises(TypeError):
                test_func(incorrect_scores_a, incorrect_scores_b)

        with self.assertRaises(TypeError):
            bonferroni_correction(incorrect_p_values)

    def test_pytorch_compatibility(self):
        """
        Test compatibility of functions with PyTorch tensors.
        """
        try:
            import torch

            # These should work
            correct1_scores_a = torch.FloatTensor(self.scores_a)
            correct1_scores_b = torch.FloatTensor(self.scores_b)
            correct2_scores_a = correct1_scores_a.unsqueeze(dim=1)
            correct2_scores_b = correct1_scores_b.unsqueeze(dim=1)

            for test_func in [bootstrap_test, permutation_test, self._aso_wrapper]:
                for scores_a, scores_b in [
                    (correct1_scores_a, correct1_scores_b),
                    (correct2_scores_a, correct2_scores_b),
                ]:
                    test_func(scores_a, scores_b)

            correct1_p_values = torch.FloatTensor(self.p_values)
            correct2_p_values = correct1_p_values.unsqueeze(dim=1)

            bonferroni_correction(correct1_p_values)
            bonferroni_correction(correct2_p_values)

            # These shouldn't
            incorrect_scores_a = torch.ones((2, 4)) * correct1_scores_a
            incorrect_scores_b = torch.ones((2, 4)) * correct1_scores_b
            incorrect_p_values = torch.ones((2, 3)) * correct1_p_values

            for test_func in [bootstrap_test, permutation_test, self._aso_wrapper]:
                with self.assertRaises(TypeError):
                    test_func(incorrect_scores_a, incorrect_scores_b)

            with self.assertRaises(TypeError):
                bonferroni_correction(incorrect_p_values)

        except ImportError:
            pass

    def test_tensorflow_compatibility(self):
        """
        Test compatibility of functions with Tensorflow tensors.
        """
        try:
            import tensorflow as tf

            # These should work
            correct1_scores_a = tf.convert_to_tensor(self.scores_a)
            correct1_scores_b = tf.convert_to_tensor(self.scores_b)
            correct2_scores_a = tf.expand_dims(correct1_scores_a, axis=1)
            correct2_scores_b = tf.expand_dims(correct1_scores_b, axis=1)

            for test_func in [bootstrap_test, permutation_test, self._aso_wrapper]:
                for scores_a, scores_b in [
                    (correct1_scores_a, correct1_scores_b),
                    (correct2_scores_a, correct2_scores_b),
                ]:
                    test_func(scores_a, scores_b)

            correct1_p_values = tf.convert_to_tensor(self.p_values)
            correct2_p_values = tf.expand_dims(correct1_p_values, axis=1)

            bonferroni_correction(correct1_p_values)
            bonferroni_correction(correct2_p_values)

            # These shouldn't
            incorrect_scores_a = tf.ones((2, 4)) * correct1_scores_a
            incorrect_scores_b = tf.ones((2, 4)) * correct1_scores_b
            incorrect_p_values = tf.ones((2, 3)) * correct1_p_values

            for test_func in [bootstrap_test, permutation_test, self._aso_wrapper]:
                with self.assertRaises(TypeError):
                    test_func(incorrect_scores_a, incorrect_scores_b)

            with self.assertRaises(TypeError):
                bonferroni_correction(incorrect_p_values)

        except ImportError:
            pass

    def test_jax_compatibility(self):
        """
        Test compatibility of functions with Jax arrays.
        """
        try:
            import jax.numpy as jnp

            # These should work
            correct1_scores_a = jnp.asarray(self.scores_a)
            correct1_scores_b = jnp.asarray(self.scores_b)
            correct2_scores_a = correct1_scores_a[..., jnp.newaxis]
            correct2_scores_b = correct1_scores_b[..., jnp.newaxis]

            for test_func in [bootstrap_test, permutation_test, self._aso_wrapper]:
                for scores_a, scores_b in [
                    (correct1_scores_a, correct1_scores_b),
                    (correct2_scores_a, correct2_scores_b),
                ]:
                    test_func(scores_a, scores_b)

            correct1_p_values = jnp.asarray(self.p_values)
            correct2_p_values = correct1_p_values[..., jnp.newaxis]

            bonferroni_correction(correct1_p_values)
            bonferroni_correction(correct2_p_values)

            # These shouldn't
            incorrect_scores_a = jnp.ones((2, 4)) * correct1_scores_a
            incorrect_scores_b = jnp.ones((2, 4)) * correct1_scores_b
            incorrect_p_values = jnp.ones((2, 3)) * correct1_p_values

            for test_func in [bootstrap_test, permutation_test, self._aso_wrapper]:
                with self.assertRaises(TypeError):
                    test_func(incorrect_scores_a, incorrect_scores_b)

            with self.assertRaises(TypeError):
                bonferroni_correction(incorrect_p_values)

        except ImportError:
            pass

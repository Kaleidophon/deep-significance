"""
Define a decorator that automatically converts python lists, tensorflow, pytorch and jax tensors into
numpy arrays for consistency.
"""

# STD
from collections import defaultdict
from functools import wraps
from typing import Callable, List, Union, Dict

# EXT
import numpy as np

# CONST & TYPES
# This defaultdict is used to apply conversions to data types defined in other frameworks (torch.Tensor /
# tf.Tensor). This is done by mapping the type of the sequence to a function converting it into an easier type.
# Thus, in the normal case, a sequence based on a Python Iterable is just kept as is.
# This trick borrowed from https://github.com/Kaleidophon/token2index/blob/master/t2i/decorators.py.
ArrayLike = Union[List[float], np.array]
ScoreCollection = Union[
    Dict[str, List[float]], Dict[str, np.array], np.array, List[List[float]]
]
CONVERSIONS = defaultdict(lambda: lambda array_like: array_like)
CONVERSIONS[list] = CONVERSIONS[tuple] = lambda array_like: np.array(array_like)
CONVERSIONS[set] = lambda array_like: np.array(list(array_like))
ALLOWED_TYPES = {list, set, tuple, np.array, np.ndarray}


def extend_type(type_: type, new_type: type) -> type:
    """ Extend a custom type which is a union of types with another type. """
    type_.__args__ = (new_type, *type_.__args__)
    return type_


# Now add conversion methods / tensors based on what packages are installed
# Ensure compatibility with Pytorch
# This is in a try/except block in case the user hasn't installed torch
try:
    import torch

    for tensor_type in (torch.FloatTensor, torch.LongTensor, torch.Tensor):
        ArrayLike = extend_type(ArrayLike, tensor_type)
        CONVERSIONS[tensor_type] = lambda t: t.detach().numpy()
        ALLOWED_TYPES.add(tensor_type)

except:
    pass

# Ensure compatibility with Tensorflow
# This is in a try/except block in case the user hasn't installed tensorflow
try:
    import tensorflow as tf

    ArrayLike = extend_type(ArrayLike, tf.Tensor)
    CONVERSIONS[tf.Tensor] = lambda t: tf.make_ndarray(t)
    ALLOWED_TYPES.add(tf.Tensor)

    from tensorflow.python.framework.ops import EagerTensor

    ArrayLike = extend_type(ArrayLike, EagerTensor)
    CONVERSIONS[EagerTensor] = lambda t: t.numpy()
    ALLOWED_TYPES.add(EagerTensor)

except:
    pass

# Ensure compatibility with Jax
# This is in a try/except block in case the user hasn't installed jax
try:
    from jax.interpreters.xla import _DeviceArray

    ArrayLike = extend_type(ArrayLike, _DeviceArray)
    CONVERSIONS[_DeviceArray] = lambda t: np.array(t)
    ALLOWED_TYPES.add(_DeviceArray)

except:
    pass


def score_conversion(func: Callable) -> Callable:
    """
    Decorator that makes sure that any sort of array containing scores is internally being converted to a numpy array.
    Supports most common Python iterables, PyTorch and Tensorflow tensors as well as Jax arrays.

    Parameters
    ----------
    func: Callable
        Function to be decorated. Should have scores_a and scores_b as first positional arguments.

    Returns
    -------
    Callable
        Decorated function.
    """

    @wraps(func)
    def with_score_conversion(
        scores_a: ArrayLike, scores_b: ArrayLike, *args, **kwargs
    ):

        # Select appropriate conversion functions
        conversion_func_a = CONVERSIONS[type(scores_a)]
        conversion_func_b = CONVERSIONS[type(scores_b)]

        # Convert to numpy arrays
        scores_a = conversion_func_a(scores_a)
        scores_b = conversion_func_b(scores_b)

        # Check dimensionality
        def _squeeze_or_exception(array: np.array, name: str) -> np.array:
            dims = len(array.shape)

            if dims > 1:
                if dims == 2 and array.shape[-1] == 1:
                    array = np.squeeze(array, axis=1)
                else:
                    raise TypeError(
                        "{} has to be one-dimensional, {} found.".format(name, dims)
                    )

            return array

        scores_a = _squeeze_or_exception(scores_a, "scores_a")
        scores_b = _squeeze_or_exception(scores_b, "scores_b")

        return func(scores_a, scores_b, *args, **kwargs)

    return with_score_conversion


def p_value_conversion(func: Callable) -> Callable:
    """
    Decorator that makes sure that any sort of array containing p-values is internally being converted to a numpy array.
    Supports most common Python iterables, PyTorch and Tensorflow tensors as well as Jax arrays.

    Parameters
    ----------
    func: Callable
        Function to be decorated. Should have scores_a and scores_b as first positional arguments.

    Returns
    -------
    Callable
        Decorated function.
    """

    @wraps(func)
    def with_p_value_conversion(p_values: ArrayLike, *args, **kwargs):

        # Select appropriate conversion functions
        conversion_func = CONVERSIONS[type(p_values)]

        # Convert to numpy arrays
        p_values = conversion_func(p_values)
        p_values = _squeeze_or_exception(p_values, "p_values")

        return func(p_values, *args, **kwargs)

    return with_p_value_conversion


def _squeeze_or_exception(array: np.array, name: str) -> np.array:
    """
    Squeeze a two-dimensional array if possible. If not, throw a TypeError.

    Parameters
    ----------
    array: np.array
        Numpy array to be squeezed.
    name: str
        Name of the array (for better error message).

    Returns
    -------
    np.array
        Squeezed, 1D Numpy array.
    """
    dims = len(array.shape)

    if dims > 1:
        if dims == 2 and array.shape[-1] == 1:
            array = np.squeeze(array, axis=1)
        else:
            raise TypeError(
                "{} has to be one-dimensional, {} found.".format(name, dims)
            )

    return array

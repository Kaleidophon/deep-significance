"""
Define a decorator that automatically converts python lists, tensorflow, pytorch and jax tensors into
numpy arrays for consistency.
"""

# STD
from functools import wraps
from typing import Callable, List, Union, Dict, Optional, TYPE_CHECKING

# EXT
import numpy as np

# Tensor types are only imported for type checkers. At runtime, the frameworks are never imported by deepsig itself:
# If a user passes e.g. a PyTorch tensor, PyTorch has already been imported anyway.
if TYPE_CHECKING:
    import jax
    import tensorflow as tf
    import torch


def _convert_torch_tensor(tensor: "torch.Tensor") -> np.ndarray:
    """
    Convert a PyTorch tensor into a numpy array. Floating point types without a numpy equivalent (like bfloat16 or
    float8 types) are converted to float64 first, which is lossless.
    """
    # Doesn't actually import anything, PyTorch has to be imported already for a tensor to exist
    import torch

    tensor = tensor.detach().cpu()

    if tensor.is_floating_point() and tensor.dtype not in (
        torch.float16,
        torch.float32,
        torch.float64,
    ):
        tensor = tensor.double()

    return tensor.numpy()


# CONST & TYPES
# This dictionary is used to apply conversions to common Python types. This is done by mapping the type of the
# sequence to a function converting it into a numpy array.
# This trick borrowed from https://github.com/Kaleidophon/token2index/blob/master/t2i/decorators.py.
CONVERSIONS = {
    list: np.array,
    tuple: np.array,
    set: lambda array_like: np.array(list(array_like)),
    np.ndarray: np.asarray,
}

# Conversions for tensors of other frameworks, based on the top-level package that their type is defined in. This way,
# frameworks never have to be imported just to check whether an object is one of their tensors.
FRAMEWORK_CONVERSIONS = {
    "torch": _convert_torch_tensor,
    "tensorflow": lambda t: t.numpy(),  # Eager tensors and variables
    "jax": np.asarray,
    "jaxlib": np.asarray,  # Concrete Jax array classes are defined in jaxlib
}

ArrayLike = Union[
    List[float], np.ndarray, "torch.Tensor", "tf.Tensor", "tf.Variable", "jax.Array"
]
ScoreCollection = Union[
    Dict[str, List[float]], Dict[str, np.ndarray], np.ndarray, List[List[float]]
]


def _get_known_conversion_func(array_like) -> Optional[Callable]:
    """
    Find the conversion function for a known type, i.e. a common Python type, a numpy array or a tensor of a supported
    framework. Parent classes are considered as well, so that e.g. subclasses of torch.Tensor are supported.

    Parameters
    ----------
    array_like: Any
        Object containing scores or p-values.

    Returns
    -------
    Optional[Callable]
        Function converting the object into a numpy array, or None if the type is not known.
    """
    for type_ in type(array_like).__mro__:
        if type_ in CONVERSIONS:
            return CONVERSIONS[type_]

        framework = type_.__module__.split(".")[0]

        if framework in FRAMEWORK_CONVERSIONS:
            return FRAMEWORK_CONVERSIONS[framework]

    return None


def get_conversion_func(array_like) -> Callable:
    """
    Select the function that converts an array-like object into a numpy array. For types that are not explicitly
    supported (e.g. pandas Series), numpy's own conversion is used.

    Parameters
    ----------
    array_like: Any
        Object containing scores or p-values.

    Returns
    -------
    Callable
        Function converting the object into a numpy array.
    """
    conversion_func = _get_known_conversion_func(array_like)

    return conversion_func if conversion_func is not None else np.asarray


def is_allowed_type(array_like) -> bool:
    """
    Check whether the type of an object (or one of its parent classes) is explicitly supported.

    Parameters
    ----------
    array_like: Any
        Object containing scores or p-values.

    Returns
    -------
    bool
        Whether the type of the object is supported.
    """
    return _get_known_conversion_func(array_like) is not None


def score_pair_conversion(func: Callable) -> Callable:
    """
    Decorator that makes sure that any sort of array containing scores is internally being converted to a numpy array.
    This decorator is specficially used for functions that that two sets of scores as their first argument.
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
    def with_score_pair_conversion(
        scores_a: ArrayLike, scores_b: ArrayLike, *args, **kwargs
    ):

        # Select appropriate conversion functions
        conversion_func_a = get_conversion_func(scores_a)
        conversion_func_b = get_conversion_func(scores_b)

        # Convert to numpy arrays
        scores_a = conversion_func_a(scores_a)
        scores_b = conversion_func_b(scores_b)

        # Check dimensionality
        def _squeeze_or_exception(array: np.ndarray, name: str) -> np.ndarray:
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

    return with_score_pair_conversion


def score_conversion(func: Callable) -> Callable:
    """
    Decorator that makes sure that any sort of array containing scores is internally being converted to a numpy array.
    In comparison to score_pair_conversion, this decorator is used for functions only using a single set of scores
    (or valuues). Supports most common Python iterables, PyTorch and Tensorflow tensors as well as Jax arrays.

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
    def with_score_conversion(scores: ArrayLike, *args, **kwargs):

        # Select appropriate conversion functions
        conversion_func = get_conversion_func(scores)

        # Convert to numpy arrays
        scores = conversion_func(scores)
        scores = _squeeze_or_exception(scores, "p_values")

        return func(scores, *args, **kwargs)

    return with_score_conversion


def _squeeze_or_exception(array: np.ndarray, name: str) -> np.ndarray:
    """
    Squeeze a two-dimensional array if possible. If not, throw a TypeError.

    Parameters
    ----------
    array: np.ndarray
        Numpy array to be squeezed.
    name: str
        Name of the array (for better error message).

    Returns
    -------
    np.ndarray
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

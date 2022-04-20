"""
Module comprising test-unrelated utility functions.
"""

# EXT
from tqdm import tqdm

# PKG
from deepsig.conversion import ScoreCollection, ALLOWED_TYPES, CONVERSIONS


def _progress_iter(high: int, progress_bar: tqdm):
    """
    This function is used when a shared progress bar is passed from multi_aso() - every time the iterator yields an
    element, the progress bar is updated by one. It essentially behaves like a simplified range() function.

    Parameters
    ----------
    high: int
        Number of elements in iterator.
    progress_bar: tqdm
        Shared progress bar.
    """
    current = 0

    while current < high:
        yield current
        current += 1
        progress_bar.update(1)


def _get_num_models(scores: ScoreCollection) -> int:
    """
    Retrieve the number of models from a ScoreCollection for multi_aso().

    Parameters
    ----------
    scores: ScoreCollection
        Collection of model scores. Should be either dictionary of model name to model scores, nested Python list,
        2D numpy or Jax array, or 2D Tensorflow or PyTorch tensor.

    Returns
    -------
    int
        Number of models.
    """
    # Python dictionary
    if isinstance(scores, dict):
        if len(scores) < 2:
            raise ValueError(
                "'scores' argument should contain at least two sets of scores, but only {} found.".format(
                    len(scores)
                )
            )

        return len(scores)

    # (Nested) python list
    elif isinstance(scores, list):
        if not isinstance(scores[0], list):
            raise TypeError(
                "'scores' argument must be nested list of scores when Python lists are used, but elements of type {} "
                "found".format(type(scores[0]).__name__)
            )

        return len(scores)

    # Numpy / Jax arrays, Tensorflow / PyTorch tensor
    elif type(scores) in ALLOWED_TYPES:
        scores = CONVERSIONS[type(scores)](scores)  # Convert to numpy array

        return scores.shape[0]

    raise TypeError(
        "Invalid type for 'scores', should be nested Python list, dict, Jax / Numpy array or Tensorflow / PyTorch "
        "tensor, '{}' found.".format(type(scores).__name__)
    )

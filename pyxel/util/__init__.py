#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Utility functions."""

import logging


import numpy as np

from typing import Callable

# flake8: noqa
from .memory import get_size, memory_usage_details
from .examples import download_examples
from .timing import time_pipeline
from .add_model import create_model
from .randomize import set_random_seed
from .image import fit_into_array, load_cropped_and_aligned_image

__all__ = [
    "convert_to_int",
    "round_convert_to_int",
    "LogFilter",
    "load_cropped_and_aligned_image",
    "set_random_seed",
    "deprecated",
    "get_dtype",
]


def round_convert_to_int(input_array: np.ndarray) -> np.ndarray:
    """Round list of floats in numpy array and convert to integers.

    Use on data before adding into DataFrame.

    :param input_array: numpy array object OR numpy array (float, int)
    :return:
    """
    array = input_array.astype(float)
    array = np.rint(array)
    array = array.astype(int)
    return array


def convert_to_int(input_array: np.ndarray) -> np.ndarray:
    """Convert numpy array to integer.

    Use on data after getting it from DataFrame.

    :param input_array: numpy array object OR numpy array (float, int)
    :return:
    """
    return input_array.astype(int)


class LogFilter(logging.Filter):
    """TBW."""

    def filter(self, record):
        """Filter or modify record."""
        if record.threadName == "MainThread" or record.threadName.endswith("-0"):
            return True

        return False


def deprecated(msg: str) -> Callable:
    """Deprecate a function.

    This decorator is based on (future ?) PEP 702.

    Examples
    --------
    >>> deprecated("Use 'new_function'")
    >>> def old_function(a, b):
    ...     return a + b
    ...

    >>> old_function.__deprecated__
    'Use new_function'
    >>> old_function(1, 2)
    DeprecationWarning: Use 'new_function'
      old_function(1, 2)
    """

    def _decorator(func: Callable) -> Callable:
        import warnings
        from functools import wraps

        @wraps(func)
        def _wrapper(*args, **kwargs):
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        # Add a new attribute '__deprecated__' to retrieve the error message (e.g. in 'ModelFunction')
        _wrapper.__deprecated__ = msg  # type: ignore
        return _wrapper

    return _decorator


def convert_unit(name: str) -> str:
    """Convert a unit name to its corresponding Unicode representation.

    Parameters
    ----------
    name : str
        A string representing a unit name.

    Returns
    -------
    str
        The unicode representation of the unit name.

    Examples
    --------
    >>> convert_unit("electron")
    'e⁻'
    """
    # Late import to speedup start-up time
    import astropy.units as u

    try:
        unit = u.Unit(name)
        return f"{unit:unicode}"
    except ValueError:
        return name


def get_dtype(bit_resolution: int) -> np.dtype:
    """Get NumPy data type based on a given bit resolution.

    Parameters
    ----------
    bit_resolution : int
        Number of bits representing the data.

    Returns
    -------
    np.dtype
        Numpy data type corresponding to the provided bit resolution.

    Raises
    ------
    ValueError
        Raised if the bit resolution does not fall within the supported range [1, 64]

    Examples
    --------
    >>> get_dtype(8)
    dtype('uint8')
    >>> get_dtype(12)
    dtype('uint16')
    """
    if 1 <= bit_resolution <= 8:
        return np.dtype(np.uint8)
    elif 9 <= bit_resolution <= 16:
        return np.dtype(np.uint16)
    elif 17 <= bit_resolution <= 32:
        return np.dtype(np.uint32)
    elif 33 <= bit_resolution <= 64:
        return np.dtype(np.uint64)
    else:
        raise ValueError(
            "Bit resolution does not fall within the supported range [1, 64]"
        )

"""Pyxel Array class."""
import typing as t  # noqa: F401

import numpy as np
from astropy.units import cds

# FRED: Is it possible to move this to `data_structure/__init__.py' ?
cds.enable()


# FRED: Does it make sense to force 'self._array' to be read-only ?
#       It could be done with:
#       ... self._array = np.array(value)
#       ... self._array.setflags(write=False)
class Array:
    """Array class."""

    EXP_TYPE = type(None)  # type: t.Type
    TYPE_LIST = ()  # type: t.Tuple[t.Type, ...]

    # FRED: Add units ?
    def __init__(self):
        """TBW."""
        # FRED: self.exp_type and self.type_list could be Class variables instead of instance variable
        #       It is more clear
        # FRED: is `self.type` needed ?

        self.type = None            # type: t.Optional[type]

        # FRED: Implement a method to initialized 'self._array' ???
        self._array = None          # type: t.Optional[np.ndarray]

    @property
    def array(self) -> np.ndarray:
        """
        Two dimensional numpy array storing the data.

        Only accepts an array with the right type and shape.
        """
        if self._array is None:
            raise ValueError("'array' is not initialized.")

        return self._array

    @array.setter
    def array(self, value: np.ndarray):
        """
        Overwrite the two dimensional numpy array storing the data.

        Only accepts an array with the right type and shape.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError('%s array should be a numpy.ndarray' % self.__class__.__name__)

        if value.dtype not in self.TYPE_LIST:
            raise TypeError('Type of %s array should be a(n) %s' %
                            (self.__class__.__name__, self.EXP_TYPE.__name__))

        if value.shape != self.array.shape:
            raise ValueError('Shape of %s array should be %s' %
                             (self.__class__.__name__, str(self.array.shape)))

        self.type = value.dtype
        self._array = value

    # # TODO: Is it necessary ? Maybe not if you implement method __array__
    # @property
    # def mean(self) -> np.ndarray:
    #     """Return mean of all pixel values."""
    #     return np.mean(self._array)
    #
    # @property
    # def std_deviation(self) -> np.ndarray:
    #     """Return standard deviation of all pixel values."""
    #     return np.std(self._array)
    #
    # @property
    # def max(self) -> np.ndarray:
    #     """Return maximum of all pixel values."""
    #     return np.max(self._array)
    #
    # @property
    # def min(self) -> np.ndarray:
    #     """Return minimum of all pixel values."""
    #     return np.min(self._array)
    #
    # @property
    # def peak_to_peak(self) -> np.ndarray:
    #     """Return peak-to-peak value of all pixel values."""
    #     return np.ptp(self._array)
    #
    # @property
    # def sum(self) -> np.ndarray:
    #     """Return sum of all pixel values."""
    #     return np.sum(self._array)

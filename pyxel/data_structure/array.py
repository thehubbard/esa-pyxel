"""Pyxel Array class."""
import numpy as np
import typing as t  # noqa: F401
from astropy.units import cds

# FRED: Is it possible to move this to `data_structure/__init__.py' ?
cds.enable()


# FRED: Does it make sense to force 'self._array' to be read-only ?
#       It could be done with:
#       ... self._array = np.array(value)
#       ... self._array.setflags(write=False)
class Array:
    """Array class."""

    # FRED: Add units ?
    def __init__(self) -> None:
        """TBW."""
        # FRED: self.exp_type and self.type_list could be Class variables instead of instance variable
        #       It is more clear

        self.type = None            # type: t.Optional[type]
        self.exp_type = None        # type: t.Optional[type]
        self.type_list = None       # type: t.Optional[t.List[type]]

        # FRED: Implement a method to initialized 'self._array' ???
        self._array = None          # type: t.Optional[np.ndarray]

    @property
    def array(self):
        """
        Two dimensional numpy array storing the data.

        Only accepts an array with the right type and shape.
        """
        return self._array

    @array.setter
    def array(self, value):
        """
        Overwrite the two dimensional numpy array storing the data.

        Only accepts an array with the right type and shape.
        """
        # FRED: This could be flattened
        if isinstance(value, np.ndarray):
            if value.dtype in self.type_list:
                if value.shape == self._array.shape:
                    self.type = value.dtype
                    self._array = value
                else:
                    raise ValueError('Shape of %s array should be %s' %
                                     (self.__class__.__name__, str(self._array.shape)))
            else:
                raise TypeError('Type of %s array should be a(n) %s' %
                                (self.__class__.__name__, self.exp_type.__name__))
        else:
            raise TypeError('%s array should be a numpy.ndarray' % self.__class__.__name__)

    # TODO: Is it necessary ? Maybe not if you implement method __array__
    @property
    def mean(self):
        """Return mean of all pixel values."""
        return np.mean(self._array)

    @property
    def std_deviation(self):
        """Return standard deviation of all pixel values."""
        return np.std(self._array)

    @property
    def max(self):
        """Return maximum of all pixel values."""
        return np.max(self._array)

    @property
    def min(self):
        """Return minimum of all pixel values."""
        return np.min(self._array)

    @property
    def peak_to_peak(self):
        """Return peak-to-peak value of all pixel values."""
        return np.ptp(self._array)

    @property
    def sum(self):
        """Return sum of all pixel values."""
        return np.sum(self._array)

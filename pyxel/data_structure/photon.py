#   --------------------------------------------------------------------------
#   Copyright 2019 SCI-FIV, ESA (European Space Agency)
#   --------------------------------------------------------------------------
"""Pyxel Photon class to generate and track photon."""

import numpy as np
from astropy.units import cds

from pyxel.data_structure.array import Array

# if TYPE_CHECKING:
#     from pyxel.detectors.geometry import Geometry

cds.enable()


class Photon(Array):
    """Photon class defining and storing information of all photon.

    Accepted array types: np.int32, np.int64, np.uint32, np.uint64, np.float16, np.float32, np.float64
    """

    EXP_TYPE = np.int
    TYPE_LIST = (np.int32, np.int64, np.uint32, np.uint64, np.float16, np.float32, np.float64)

    # FRED: Replace by '__init__(self, new_array: np.ndarray)' ???
    # def __init__(self, geo: "Geometry"):
    #     """TBW.
    #
    #     :param geo:
    #     :param array:
    #     """
    #     super().__init__()                  # TODO: add unit (ph)

    # FRED: This could be done in '__init__'
    def new_array(self, new_array: np.ndarray) -> None:
        """TBW.

        :param new_array:
        """
        if not isinstance(new_array, np.ndarray):
            raise TypeError('%s array should be a numpy.ndarray' % self.__class__.__name__)

        if new_array.dtype not in self.TYPE_LIST:
            raise TypeError('Type of %s array should be a(n) %s' %
                            (self.__class__.__name__, self.EXP_TYPE.__name__))

        self._array = new_array
        self.type = new_array.dtype  # FRED: Where is it used ?

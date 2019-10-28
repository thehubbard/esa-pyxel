"""Pyxel Pixel class."""
from typing import TYPE_CHECKING

import numpy as np
from astropy.units import cds

from pyxel.data_structure.array import Array

if TYPE_CHECKING:
    from pyxel.detectors.geometry import Geometry

cds.enable()


class Pixel(Array):
    """Pixel class defining and storing information of charge packets within pixel.

    Accepted array types: np.int32, np.int64, np.uint32, np.uint64, np.float16, np.float32, np.float64.
    """

    EXP_TYPE = np.int
    TYPE_LIST = (np.int32, np.int64, np.uint32, np.uint64, np.float16, np.float32, np.float64)

    def __init__(self, geo: "Geometry"):
        """TBW.

        :param geo:
        """
        new_array = np.zeros((geo.row, geo.col), dtype=self.EXP_TYPE)

        super().__init__(new_array)

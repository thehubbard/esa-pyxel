"""Pyxel Signal class."""
from typing import TYPE_CHECKING

import numpy as np
from astropy.units import cds
from pyxel.data_structure.array import Array

if TYPE_CHECKING:
    from pyxel.detectors.geometry import Geometry

cds.enable()


class Signal(Array):
    """Signal class defining and storing information of detector signal.

    Accepted array types: np.float16, np.float32, np.float64.
    """

    EXP_TYPE = np.float
    TYPE_LIST = (np.float16, np.float32, np.float64)

    def __init__(self, geo: "Geometry"):
        """TBW.

        :param geo:
        """
        new_array = np.zeros((geo.row, geo.col), dtype=self.EXP_TYPE)

        super().__init__(new_array)  # TODO: add unit (V)

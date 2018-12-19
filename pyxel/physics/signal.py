#   --------------------------------------------------------------------------
#   Copyright 2018 SCI-FIV, ESA (European Space Agency)
#   --------------------------------------------------------------------------
"""Pyxel Signal class."""
import numpy as np
from astropy.units import cds
from pyxel.detectors.geometry import Geometry
from pyxel.physics.array import Array

cds.enable()


class Signal(Array):
    """Signal class defining and storing information of detector signal."""

    def __init__(self, geo: Geometry) -> None:
        """TBW.

        :param geo:
        """
        super().__init__()
        self.exp_type = np.float
        self.type_list = [np.float16, np.float32, np.float64]
        self._array = np.zeros((geo.row, geo.col), dtype=self.exp_type)

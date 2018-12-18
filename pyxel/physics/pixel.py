#   --------------------------------------------------------------------------
#   Copyright 2018 SCI-FIV, ESA (European Space Agency)
#   --------------------------------------------------------------------------
"""Pyxel Pixel class to store and transfer charge packets inside detector."""
import numpy as np
from astropy.units import cds
from pyxel.detectors.geometry import Geometry
from pyxel.physics.array import Array

cds.enable()


class Pixel(Array):
    """Pixel class defining and storing information of charge packets.

    Pixel properties stored are: charge, position, lost charge.
    """

    def __init__(self, geo: Geometry) -> None:
        """TBW.

        :param geo:
        """
        super().__init__()
        self.exp_type = np.int
        self.type_list = [np.int, np.int16, np.int32, np.int64]        # uint16, uint64 ??
        self._array = np.zeros((geo.row, geo.col), dtype=self.exp_type)

#   Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#  #
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this Pyxel package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
"""Pyxel Photon 3D class to generate and track 3D photon."""

import warnings
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from pyxel.data_structure import Array

if TYPE_CHECKING:
    from pyxel.detectors import Geometry


class Photon3d:
    """Photon class defining and storing information of all 3d photon.
    The dimensions are wavelength, y and x.

    Accepted array types: ``np.float16``, ``np.float32``, ``np.float64``
    """

    EXP_TYPE = float
    TYPE_LIST = (
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
    )
    NAME = "Photon3d"
    UNIT = "Ph/nm"  # ?

    def __init__(self, array):
        assert array.ndim == 3

        assert array.dims == ("wavelength", "y", "x")

        self._array = array

    @property
    def array(self) -> xr.DataArray:
        """Three-dimensional numpy array storing the data.

        Only accepts an DataArray with the right type and shape.
        """
        return self._array

    @array.setter
    def array(self, value: xr.DataArray) -> None:
        """Overwrite the 3-dimensional DataArray storing the data.

        Only accepts an array with the right type and shape.
        """

        assert value.ndim == 3

        assert value.dims == ("wavelength", "y", "x")

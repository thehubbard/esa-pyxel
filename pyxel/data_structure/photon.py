#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Pyxel Photon class to generate and track photon."""

import warnings
from typing import TYPE_CHECKING

import numpy as np

from pyxel.data_structure import Array

if TYPE_CHECKING:
    from pyxel.detectors import Geometry


class Photon(Array):
    """Photon class defining and storing information of all photon.

    Accepted array types: ``np.int32``, ``np.int64``, ``np.uint32``, ``np.uint64``,
    ``np.float16``, ``np.float32``, ``np.float64``
    """

    # TODO: add unit (ph)
    EXP_TYPE = float
    TYPE_LIST = (
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
    )
    NAME = "Photon"
    UNIT = "Ph"

    def __init__(self, geo: "Geometry"):
        super().__init__(shape=(geo.row, geo.col))

    def _validate(self, value: np.ndarray) -> None:
        """Check that values in array are all positive."""
        if np.any(value < 0):
            value[value < 0] = 0.0
            warnings.warn(
                "Trying to set negative values in the Photon array! Negative values"
                " clipped to 0.",
                stacklevel=2,
            )
        super()._validate(value)

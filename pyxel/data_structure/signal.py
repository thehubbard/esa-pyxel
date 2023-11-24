#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Pyxel Signal class."""

from typing import TYPE_CHECKING

import numpy as np

from pyxel.data_structure import Array

if TYPE_CHECKING:
    from pyxel.detectors import Geometry


class Signal(Array):
    """Signal class defining and storing information of detector signal.

    Accepted array types: ``np.float16``, ``np.float32``, ``np.float64``.
    """

    TYPE_LIST = (np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64))
    NAME = "Signal"
    UNIT = "Volt"

    def __init__(self, geo: "Geometry"):
        super().__init__(shape=(geo.row, geo.col))

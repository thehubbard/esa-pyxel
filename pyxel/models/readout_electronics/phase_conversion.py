#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Readout electronics model."""

import typing as t

from pyxel.detectors import MKID, Detector
from pyxel.models.readout_electronics.util import apply_gain_adc


def simple_phase_conversion(detector: MKID, phase_conversion: float = 1.0) -> None:
    """Create an image array from phase array.

    Parameters
    ----------
    detector: MKID
        Pyxel :term:`MKID` detector object.
    phase_conversion : float
        Phase conversion factor
    """
    if not isinstance(detector, MKID):
        raise TypeError("Expecting a MKID object for the detector.")

    detector.image.array = phase_conversion * detector.phase.array

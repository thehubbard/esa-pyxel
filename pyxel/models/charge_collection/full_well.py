#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Pyxel full well models."""
from pyxel.detectors import Detector
import typing as t
import numpy as np


def apply_full_well_capacity(array: np.ndarray, fwc: int) -> np.ndarray:
    """Apply full well capacity to an array.

    Parameters
    ----------
    array: ndarray
    fwc: int

    Returns
    -------
    output: ndarray
    """
    output = np.copy(array)
    output[output > fwc] = fwc
    return output


def simple_full_well(detector: Detector, fwc: t.Optional[int] = None) -> None:
    """Limit the amount of charge in pixel due to full well capacity.

    Parameters
    ----------
    detector: Detector
    fwc: int
    """
    if not fwc:
        fwc = detector.characteristics.fwc

    if fwc < 0:
        raise ValueError("Full well capacity should be a positive number.")

    charge_array = apply_full_well_capacity(array=detector.pixel.array, fwc=fwc)

    detector.pixel.array = charge_array

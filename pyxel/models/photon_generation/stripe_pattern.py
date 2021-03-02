#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#

import typing as t

import numpy as np

from pyxel.data_structure import Photon
from pyxel.detectors import
import skimage.transform as tr

if t.TYPE_CHECKING:
    from pyxel.detectors import Detector


def square_signal(n: int, lw: int, startwith: int =0):
    """TBW."""
    if lw > n // 2:
        raise ValueError("Line too wide.")
    start = [startwith] * lw
    second = [1 - startwith] * lw
    pair = start + second
    num = n // len(pair)
    out = pair * num
    return out


def pattern(detector_shape, period: int =2, multiplier=1, startwith=0, angle=0):
    """TBW."""
    x, y = detector_shape

    n, m = x*4, y*4

    sx = slice(n//2-x//2, n//2+x//2)
    sy = slice(m // 2 - y // 2, m // 2 + y // 2)

    signal = square_signal(n, period, startwith)
    signal = np.array(signal + ([1] * (n - len(signal))))[::-1]

    out = np.ones((n,m))

    for i in range(m):
        out[:, i] = signal

    out = multiplier * out

    if angle:
        out = tr.rotate(out, angle=angle)

    return out[sx, sy]


def stripe_pattern(
    detector: Detector, period: int = 10, level: float = 1., angle:int =0, startwith: int = 0):
    """TBW."""

    if period<2:
        raise ValueError("Cant set a period smaller than 2.")
    elif (period%2) != 0:
        raise ValueError("Period should be a multiple of 2.")

    photon_array = pattern(
        detector_shape=(detector.geometry.row, detector.geometry.col),
        multiplier=level,
        period=period,
        startwith=startwith,
        angle=angle
    )

    detector.photon = Photon(photon_array)
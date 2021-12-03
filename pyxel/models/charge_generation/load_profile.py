#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Simple model to load charge profiles."""

import typing as t
from functools import lru_cache
from pathlib import Path

import numpy as np

from pyxel.detectors import Detector, Geometry
from pyxel.inputs import load_image
from pyxel.util import fit_into_array
from typing_extensions import Literal


@lru_cache(maxsize=128)  # One must add parameter 'maxsize' for Python 3.7
def load_charge_from_file(
    shape: t.Tuple[int, int],
    filename: t.Union[str, Path],
    position: t.Tuple[int, int] = (0, 0),
    align: t.Optional[
        Literal["center", "top_left", "top_right", "bottom_left", "bottom_right"]
    ] = None,
) -> np.ndarray:
    """Create charges from a charge profile file.

    Parameters
    ----------
    shape: tuple
    filename: str or Path
    position: tuple
        Indices of starting row and column, used when fitting charge to detector.
    align: Literal
        Keyword to align the charge to detector. Can be any from:
        ("center", "top_left", "top_right", "bottom_left", "bottom_right")

    Returns
    -------
    ndarray
    """
    # Load 2d charge profile (which can be smaller or
    #                         larger in dimensions than detector imaging area)
    charges_from_file_2d = load_image(filename)

    cropped_and_aligned_charge = fit_into_array(
        array=charges_from_file_2d, output_shape=shape, relative_position=position, align=align
    )  # type: np.ndarray

    return cropped_and_aligned_charge


def charge_profile(
    detector: Detector,
    filename: t.Union[str, Path],
    position: t.Tuple[int, int] = (0, 0),
    align: t.Optional[
        Literal["center", "top_left", "top_right", "bottom_left", "bottom_right"]
    ] = None,
    time_scale: float = 1.0,
) -> None:
    """Load charge profile from txt file for detector, mostly for but not limited to CCDs.

    Parameters
    ----------
    detector : Detector
        Pyxel Detector object.
    filename : str or Path
        File path.
    position: tuple
        Indices of starting row and column, used when fitting charge to detector.
    align: Literal
        Keyword to align the charge to detector. Can be any from:
        ("center", "top_left", "top_right", "bottom_left", "bottom_right")
    time_scale: float
        Time scale of the input charge, default is 1 second. 0.001 would be ms.
    """
    geo = detector.geometry  # type: Geometry

    # Load charge profile as numpy array.
    charges = load_charge_from_file(
        shape=(geo.row, geo.col),
        filename=filename,
        position=position,
        align=align,
    )  # type: np.ndarray

    charges *= detector.time_step / time_scale

    # Add charges in 'detector'
    detector.charge.add_charge_array(charges)

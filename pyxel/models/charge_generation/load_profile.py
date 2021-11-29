#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Simple model to load charge profiles."""

import logging
import typing as t
from pathlib import Path

import numpy as np

from pyxel.detectors import Detector, Geometry

# TODO: more documentation, private function


def load_charge_from_file(
    num_rows: int,
    num_cols: int,
    txt_file: str,
    profile_position_y: int,
    profile_position_x: int,
    fit_profile_to_det: bool = False,
) -> np.ndarray:
    """Create charges from a charge profile file."""
    # All pixels has zero charge by default
    detector_charge_2d = np.zeros((num_rows, num_cols))

    # Load 2d charge profile (which can be smaller or
    #                         larger in dimensions than detector imaging area)
    full_path = Path(txt_file).resolve()
    charges_from_file_2d = np.loadtxt(str(full_path), ndmin=2)  # type: np.ndarray
    # TODO: use pyxel function load_table?

    if fit_profile_to_det:
        # Crop 2d charge profile, so it is not larger in dimensions than detector imaging area)
        charges_from_file_2d = charges_from_file_2d[
            slice(0, num_rows), slice(0, num_cols)
        ]

    profile_rows, profile_cols = charges_from_file_2d.shape

    detector_charge_2d[
        slice(profile_position_y, profile_position_y + profile_rows),
        slice(profile_position_x, profile_position_x + profile_cols),
    ] = charges_from_file_2d

    return detector_charge_2d


def charge_profile(
    detector: Detector,
    txt_file: t.Union[str, Path],
    fit_profile_to_det: bool = False,
    profile_position: t.Optional[t.Tuple[int, int]] = None,
) -> None:
    """Load charge profile from txt file for detector, mostly for but not limited to CCDs.

    Parameters
    ----------
    detector : Detector
        Pyxel Detector object.
    txt_file : str or Path
        File path.
    fit_profile_to_det : bool
    profile_position : list
    """
    logging.info("")

    if profile_position is None:
        profile_position_y = 0  # type: int
        profile_position_x = 0  # type: int
    else:
        profile_position_y, profile_position_x = profile_position

    geo = detector.geometry  # type: Geometry

    # Load charge profile as numpy array.
    charges = load_charge_from_file(
        num_rows=geo.row,
        num_cols=geo.col,
        txt_file=txt_file,
        profile_position_y=profile_position_y,
        profile_position_x=profile_position_x,
        fit_profile_to_det=fit_profile_to_det,
    )  # type: np.ndarray

    # Add charges in 'detector'
    detector.charge.add_charge_array(charges)

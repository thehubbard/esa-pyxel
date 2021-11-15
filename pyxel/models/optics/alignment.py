#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Pyxel photon generator models."""
import numpy as np

from pyxel.detectors import Detector


def apply_alignment(
    data_2d: np.ndarray,
    target_rows: int,
    target_cols: int,
) -> np.ndarray:
    """

    Parameters
    ----------
    data_2d : array
    target_rows : int
    target_cols : int

    Returns
    -------
    array
        An aligned 2D array.
    """
    rows, cols = data_2d.shape

    row0 = int((rows - target_rows) / 2)
    col0 = int((cols - target_cols) / 2)

    if row0 < 0 or col0 < 0:
        raise ValueError

    aligned_data_2d = data_2d[slice(row0, row0 + rows), slice(col0, col0 + cols)]

    return aligned_data_2d


def alignment(detector: Detector) -> None:
    """Optical alignment.

    Parameters
    ----------
    detector: Detector
        Pyxel Detector object.
    """
    geo = detector.geometry

    aligned_optical_image = apply_alignment(
        data_2d=detector.photon.array,
        target_rows=geo.row,
        target_cols=geo.col,
    )

    detector.photon.array = aligned_optical_image

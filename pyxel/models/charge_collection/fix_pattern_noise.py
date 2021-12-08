#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Fix pattern noise model."""

import typing as t
from pathlib import Path

import numpy as np

from pyxel.detectors import Detector, Geometry
from pyxel.inputs import load_image
from pyxel.util import temporary_random_state

# TODO: use built-in pyxel functions for loading files, documentation, docstring, saving to file??, private function


def create_fix_pattern_noise(num_rows: int, num_cols: int) -> np.ndarray:
    """Create a fix pattern noise.

    Parameters
    ----------
    num_rows : int
    num_cols : int

    Returns
    -------
    ndarray
        A 2D array with the noise. Unit: electron
    """
    pnu_2d = np.random.normal(loc=1.0, scale=0.03, size=(num_rows, num_cols))

    return pnu_2d


def load_fix_pattern_noise(num_rows: int, num_cols: int, filename: Path) -> np.ndarray:
    """Load a fix pattern noise from a filename.

    Parameters
    ----------
    num_rows : int
    num_cols : int
    filename : str or Path
        Path used to load a fix pattern noise.

    Returns
    -------
    ndarray
        A 2D array. Unit: electron
    """
    pnu_2d = load_image(filename)  # type: np.ndarray

    return pnu_2d.reshape((num_rows, num_cols))


@temporary_random_state
def fix_pattern_noise(
    detector: Detector,
    pixel_non_uniformity: t.Union[str, Path, None] = None,
    seed: t.Optional[int] = None,
) -> None:
    """Add fix pattern noise caused by pixel non-uniformity during charge collection.

    Parameters
    ----------
    detector : Detector
        Pyxel Detector object.
    pixel_non_uniformity : str or Path
        path to an ascii file with and array.
    seed: int, optional

    Raises
    ------
    FileNotFoundError
        If filename 'pixel_non_uniformity' is provided but does not exist.
    """
    # Validation
    if pixel_non_uniformity and not Path(pixel_non_uniformity).exists():
        raise FileNotFoundError(
            f"Cannot find filename 'pixel_non_uniformity': '{pixel_non_uniformity}"
        )

    geo = detector.geometry  # type: Geometry

    if pixel_non_uniformity:
        pnu_2d = load_fix_pattern_noise(
            num_rows=geo.row,
            num_cols=geo.col,
            filename=Path(pixel_non_uniformity),
        )  # type: np.ndarray
    else:
        pnu_2d = create_fix_pattern_noise(num_rows=geo.row, num_cols=geo.col)

    detector.pixel.array *= pnu_2d

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Fix pattern noise model."""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from typing_extensions import Literal

from pyxel.detectors import Detector, Geometry
from pyxel.util import load_cropped_and_aligned_image


def fix_pattern_noise(
    detector: Detector,
    filename: Optional[str, Path] = None,
    fixed_pattern_noise_factor: Optional[float] = None,
    position: Tuple[int, int] = (0, 0),
    align: Optional[
        Literal["center", "top_left", "top_right", "bottom_left", "bottom_right"]
    ] = None,
) -> None:
    """Add fixed pattern noise caused by pixel non-uniformity during charge collection.

    Parameters
    ----------
    fixed_pattern_noise_factor: float
        Fixed pattern noise factor.
    detector : Detector
        Pyxel Detector object.
    filename : str or Path
        Path to a file with an array or an image.
    position : tuple
        Indices of starting row and column, used when fitting noise to detector.
    align : Literal
        Keyword to align the noise to detector. Can be any from:
        ("center", "top_left", "top_right", "bottom_left", "bottom_right")
    """
    geo: Geometry = detector.geometry
    position_y, position_x = position
    shape = geo.shape
    qe = detector.characteristics.quantum_efficiency

    if filename is not None:
        # Load charge profile as numpy array.
        prnu_2d = load_cropped_and_aligned_image(
            shape=(geo.row, geo.col),
            filename=filename,
            position_x=position_x,
            position_y=position_y,
            align=align,
        )  # type: np.ndarray

    else:
        if fixed_pattern_noise_factor is not None:
            prnu_2d = np.ones(shape) * qe * fixed_pattern_noise_factor
            prnu_sigma = qe * fixed_pattern_noise_factor
            prnu_2d = prnu_2d * (1 + np.random.lognormal(sigma=prnu_sigma, size=shape))
        else:
            raise ValueError(
                "Either filename or fixed_pattern_noise_factor has to be defined."
            )

    detector.pixel.array *= prnu_2d

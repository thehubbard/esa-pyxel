#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Pyxel photon generator models."""
import typing as t
from pathlib import Path

import numpy as np

import pyxel
from pyxel.detectors import Detector


def load_image_from_file(
    image_file: str,
    shape: t.Tuple[int, int],
    position: t.Tuple[int, int] = (0, 0),
    fit_image_to_det: bool = False,
) -> np.ndarray:
    """Load image from file and fit to detector shape.

    Parameters
    ----------
    shape: tuple
        Detector shape.
    image_file: str
        Path to image file.
    fit_image_to_det: bool
        Fitting image to given shape.
    position: tuple
        Indices of starting row and column, used when fitting image to detector.

    Returns
    -------
    image: np.ndarray
    """
    filename = Path(image_file).expanduser().resolve()

    if not Path(filename).exists():
        raise FileNotFoundError(f"Image file '{filename}' does not exist !")

    image = pyxel.load_image(filename)  # type: np.ndarray

    row, col = shape

    # TODO: create tests for this part, see issue #337
    if fit_image_to_det:
        position_y, position_x = position

        image = image[
            slice(position_y, position_y + row),
            slice(position_x, position_x + col),
        ]

    return image


def load_image(
    detector: Detector,
    image_file: str,
    fit_image_to_det: bool = False,
    position: t.Tuple[int, int] = (0, 0),
    convert_to_photons: bool = False,
    multiplier: float = 1.0,
    time_scale: float = 1.0,
) -> None:
    r"""Load FITS file as a numpy array and add to the detector as input image.

    Parameters
    ----------
    detector: Detector
    image_file: str
        Path to image file.
    fit_image_to_det: bool
        Fitting image to detector shape (Geometry.row, Geometry.col).
    position: tuple
        Indices of starting row and column, used when fitting image to detector.
    convert_to_photons: bool
        If ``True``, the model converts the values of loaded image array from ADU to
        photon numbers for each pixel using the Photon Transfer Function:
        :math:`PTF = QE \cdot \eta \cdot S_{v} \cdot amp \cdot a_{1} \cdot a_{2}`.
    multiplier: float
        Multiply photon array level with a custom number.
    time_scale: float
        Time scale of the photon flux, default is 1 second. 0.001 would be ms.
    """

    shape = (detector.geometry.row, detector.geometry.col)

    image = load_image_from_file(
        image_file=image_file,
        shape=shape,
        fit_image_to_det=fit_image_to_det,
        position=position,
    )

    detector.input_image = image
    photon_array = image

    if convert_to_photons:
        cht = detector.characteristics
        photon_array = photon_array / (
            cht.qe * cht.eta * cht.sv * cht.amp * cht.a1 * cht.a2
        )

    photon_array = photon_array * (detector.time_step / time_scale) * multiplier

    try:
        detector.photon.array += photon_array
    except ValueError as ex:
        raise ValueError("Shapes of arrays do not match") from ex

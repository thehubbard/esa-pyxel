#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Pyxel photon generator models."""
import typing as t

import numpy as np
from typing_extensions import Literal

import pyxel
from pyxel.detectors import Detector
from pyxel.util import fit_into_array


def load_image_from_file(
    filename: str,
    shape: t.Tuple[int, int],
    position: t.Tuple[int, int] = (0, 0),
    align: t.Optional[
        Literal["center", "top_left", "top_right", "bottom_left", "bottom_right"]
    ] = None,
) -> np.ndarray:
    """Load image from file and fit to detector shape.

    Parameters
    ----------
    shape: tuple
        Detector shape.
    filename: str
        Path to image file.
    position: tuple
        Indices of starting row and column, used when fitting image to detector.
    align: Literal
        Keyword to align the image to detector. Can be any from:
        ("center", "top_left", "top_right", "bottom_left", "bottom_right")

    Returns
    -------
    image: ndarray
    """
    image = pyxel.load_image(filename)  # type: np.ndarray

    cropped_and_aligned_image = fit_into_array(
        array=image, output_shape=shape, relative_position=position, align=align
    )  # type: np.ndarray

    return cropped_and_aligned_image


def load_image(
    detector: Detector,
    image_file: str,
    position: t.Tuple[int, int] = (0, 0),
    align: t.Optional[
        Literal["center", "top_left", "top_right", "bottom_left", "bottom_right"]
    ] = None,
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
    position: tuple
        Indices of starting row and column, used when fitting image to detector.
    align: Literal
        Keyword to align the image to detector. Can be any from:
        ("center", "top_left", "top_right", "bottom_left", "bottom_right")
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
        filename=image_file,
        shape=shape,
        align=align,
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

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""POPPY (Physical Optics Propagation in Python) model wrapper.

It calculates the optical Point Spread Function of an optical system and applies the convolution.

Documentation:
https://poppy-optics.readthedocs.io/en/stable/index.html

See details about POPPY Optical Element classes:
https://poppy-optics.readthedocs.io/en/stable/available_optics.html

Supported optical elements:

- ``CircularAperture``
- ``SquareAperture``
- ``RectangularAperture``
- ``HexagonAperture``
- ``MultiHexagonalAperture``
- ``ThinLens``
- ``SecondaryObscuration``
- ``ZernikeWFE``
- ``SineWaveWFE``

.. code-block:: yaml

  # YAML config: arguments of POPPY optical_system
  optical_system:
  - item: CircularAperture
    radius: 1.5
  - item: ThinLens
    radius: 1.2
    nwaves: 1
  - item: ZernikeWFE
    radius: 0.8
    coefficients: [0.1e-6, 3.e-6, -3.e-6, 1.e-6, -7.e-7, 0.4e-6, -2.e-6]
    aperture_stop: false
"""

import logging
import typing as t
from dataclasses import dataclass

import numpy as np
from astropy.convolution import convolve_fft
from astropy.io import fits

from pyxel.detectors import Detector

try:
    import poppy as op

    WITH_POPPY: bool = True
except ImportError:
    WITH_POPPY = False


@dataclass
class CircularAperture:
    radius: float


@dataclass
class ThinLens:
    nwaves: float
    radius: float


@dataclass
class SquareAperture:
    size: int


@dataclass
class RectangleAperture:
    width: float
    height: float


@dataclass
class HexagonAperture:
    side: float


@dataclass
class MultiHexagonalAperture:
    side: float
    rings: int
    gap: float


@dataclass
class SecondaryObscuration:
    secondary_radius: float
    n_supports: int
    support_width: float


@dataclass
class ZernikeWFE:
    radius: float
    coefficients: t.Sequence[float]
    aperture_stop: float


@dataclass
class SineWaveWFE:
    spatialfreq: float
    amplitude: float
    rotation: float


# Define a type alias
OpticalParameter = t.Union[
    CircularAperture,
    ThinLens,
    SquareAperture,
    RectangleAperture,
    HexagonAperture,
    MultiHexagonalAperture,
    SecondaryObscuration,
    ZernikeWFE,
    SineWaveWFE,
]


def create_optical_parameter(dct: t.Mapping) -> OpticalParameter:
    if dct["item"] == "CircularAperture":
        return CircularAperture(radius=dct["radius"])

    elif dct["item"] == "ThinLens":
        return ThinLens(nwaves=dct["nwaves"], radius=dct["radius"])

    elif dct["item"] == "SquareAperture":
        return SquareAperture(size=dct["size"])

    elif dct["item"] == "RectangularAperture":
        return RectangleAperture(width=dct["width"], height=dct["height"])

    elif dct["item"] == "HexagonAperture":
        return HexagonAperture(side=dct["side"])

    elif dct["item"] == "MultiHexagonalAperture":
        return MultiHexagonalAperture(
            side=dct["side"],
            rings=dct["rings"],
            gap=dct["gap"],
        )  # cm

    elif dct["item"] == "SecondaryObscuration":
        return SecondaryObscuration(
            secondary_radius=dct["secondary_radius"],
            n_supports=dct["n_supports"],
            support_width=dct["support_width"],
        )  # cm

    elif dct["item"] == "ZernikeWFE":
        return ZernikeWFE(
            radius=dct["radius"],
            coefficients=dct["coefficients"],  # list of floats
            aperture_stop=dct["aperture_stop"],
        )  # bool

    elif dct["item"] == "SineWaveWFE":
        return SineWaveWFE(
            spatialfreq=dct["spatialfreq"],  # 1/m
            amplitude=dct["amplitude"],  # um
            rotation=dct["rotation"],
        )
    else:
        raise NotImplementedError


def create_optical_parameters(
    optical_system: t.Sequence[t.Mapping],
) -> t.Sequence[OpticalParameter]:
    return [create_optical_parameter(dct) for dct in optical_system]


def create_optical_item(
    param: OpticalParameter, wavelength: float
) -> op.OpticalElement:
    if isinstance(param, CircularAperture):
        return op.CircularAperture(radius=param.radius)

    elif isinstance(param, ThinLens):
        return op.ThinLens(
            nwaves=param.nwaves,
            reference_wavelength=wavelength,
            radius=param.radius,
        )

    elif isinstance(param, SquareAperture):
        return op.SquareAperture(size=param.size)

    elif isinstance(param, RectangleAperture):
        return op.RectangleAperture(width=param.width, height=param.height)

    elif isinstance(param, HexagonAperture):
        return op.HexagonAperture(side=param.side)

    elif isinstance(param, MultiHexagonalAperture):
        return op.MultiHexagonAperture(
            side=param.side, rings=param.rings, gap=param.gap
        )

    elif isinstance(param, SecondaryObscuration):
        return op.SecondaryObscuration(
            secondary_radius=param.secondary_radius,
            n_supports=param.n_supports,
            support_width=param.support_width,
        )

    elif isinstance(param, ZernikeWFE):
        return op.ZernikeWFE(
            radius=param.radius,
            coefficients=param.coefficients,
            aperture_stop=param.aperture_stop,
        )

    elif isinstance(param, SineWaveWFE):
        return op.SineWaveWFE(
            spatialfreq=param.spatialfreq,
            amplitude=param.amplitude,
            rotation=param.rotation,
        )
    else:
        raise NotImplementedError


def calc_psf(
    wavelength: float,
    fov_arcsec: float,
    pixelscale: float,
    optical_parameters: t.Sequence[OpticalParameter],
) -> t.Tuple[t.Sequence[fits.hdu.image.PrimaryHDU], t.Sequence["op.Wavefront"]]:
    """Calculate the point spread function for the given optical system and optionally display the psf.

    Parameters
    ----------
    wavelength: float
        Wavelength of incoming light in meters.
    fov_arcsec: float, optional
        Field Of View on detector plane in arcsec.
    pixelscale: float
        Pixel scale on detector plane (arcsec/pixel).
        Defines sampling resolution of PSF.
    optical_parameters:
        List of optical parameters before detector with their specific arguments.

    Returns
    -------
    psf: Sequence of FITS and sequence of Wavefront
        Tuple of lists containing the psf and intermediate wavefronts.
    """
    if not WITH_POPPY:
        raise ImportError(
            "Missing optional package 'poppy'.\n"
            "Please install it with 'pip install pyxel-sim[model]' "
            "or 'pip install pyxel-sim[all]'"
        )

    # Create the optical element(s)
    osys = op.OpticalSystem(npix=1000)  # default: 1024

    for param in optical_parameters:  # type: OpticalParameter
        element = create_optical_item(
            param=param,
            wavelength=wavelength,
        )  # type: op.OpticalElement

        osys.add_pupil(element)

    osys.add_detector(
        pixelscale=pixelscale,
        fov_arcsec=fov_arcsec,
    )

    # Calculate a monochromatic PSF
    output_fits, wavefronts = osys.calc_psf(
        wavelength=wavelength,
        return_intermediates=True,
        normalize="last",
    )  # type: t.Sequence[fits.hdu.image.PrimaryHDU], t.Sequence[op.Wavefront]

    return output_fits, wavefronts


def apply_convolution(data_2d: np.ndarray, kernel_2d: np.ndarray) -> np.ndarray:
    """Convolve an array

    Parameters
    ----------
    data_2d : array
        2D Array to be convolved with kernel_2d.
    kernel_2d : array
        The con
    Returns
    -------
    array

    """
    mean = np.mean(data_2d)

    array_2d = convolve_fft(
        data_2d,
        kernel=kernel_2d,
        boundary="fill",
        fill_value=mean,
    )

    return array_2d


def optical_psf(
    detector: Detector,
    wavelength: float,
    fov_arcsec: float,
    pixelscale: float,
    optical_system: t.Sequence[t.Mapping],
) -> None:
    """Model function for poppy optics model: convolve photon array with psf.

    Parameters
    ----------
    detector: Detector
        Pyxel Detector object.
    wavelength: float
        Wavelength of incoming light in meters.
    fov_arcsec: float, optional
        Field Of View on detector plane in arcsec.
    pixelscale: float
        Pixel scale on detector plane (arcsec/pixel).
        Defines sampling resolution of PSF.
    optical_system:
        List of optical elements before detector with their specific arguments.
    """

    logging.getLogger("poppy").setLevel(
        logging.WARNING
    )  # TODO: Fix this. See issue #81

    # Convert 'optical_system' to 'optical_parameters'
    optical_parameters = [
        create_optical_parameter(dct) for dct in optical_system
    ]  # type: t.Sequence[OpticalParameter]

    # Get a Point Spread Function
    images, wavefronts = calc_psf(
        wavelength=wavelength,
        fov_arcsec=fov_arcsec,
        pixelscale=pixelscale,
        optical_parameters=optical_parameters,
    )

    # Extract 'first_image'
    first_image, *other_images = images

    # Convolution
    new_array_2d = apply_convolution(
        data_2d=detector.photon.array,
        kernel_2d=first_image.data,
    )  # type: np.ndarray

    detector.photon.array = new_array_2d

#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Poppy model."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
from astropy.convolution import convolve_fft
from astropy.io import fits

from pyxel.detectors import Detector

try:
    import poppy as op

    WITH_POPPY: bool = True
except ModuleNotFoundError:
    WITH_POPPY = False


@dataclass
class CircularAperture:
    """Parameters for an ideal circular pupil aperture.

    Parameters
    ----------
    radius : float
        Radius of the pupil, in meters.
    """

    radius: float


@dataclass
class ThinLens:
    """Parameters for an idealized thin lens.

    Parameters
    ----------
    nwaves : float
        The number of waves of defocus, peak to valley.
    radius : float
        Pupil radius, in meters, over which the Zernike defocus term should be computed
        such that rho = 1 at r = `radius`.
    """

    nwaves: float
    radius: float


@dataclass
class NewThinLens:
    """Parameters for an idealized thin lens.

    Parameters
    ----------
    nwaves : float
        The number of waves of defocus, peak to valley.
    radius : float
        Pupil radius, in meters, over which the Zernike defocus term should be computed
        such that rho = 1 at r = `radius`.
    reference_wavelength : float
        Wavelength, in meters, at which that number of waves of defocus is specified.
    """

    nwaves: float
    radius: float
    reference_wavelength: Optional[float] = None


@dataclass
class SquareAperture:
    """Parameters for an ideal square pupil aperture.

    Parameters
    ----------
    size : float
        side length of the square, in meters.
    """

    size: float


@dataclass
class RectangleAperture:
    """Parameters for an ideal rectangular pupil aperture.

    Parameters
    ----------
    width : float
        width of the rectangle, in meters.
    height : float
        height of the rectangle, in meters.
    """

    width: float
    height: float


@dataclass
class HexagonAperture:
    """Parameters for an ideal hexagonal pupil aperture.

    Parameters
    ----------
    side : float
        side length (and/or radius) of hexagon, in meters.
    """

    side: float


@dataclass
class MultiHexagonalAperture:
    """Parameters for an hexagonaly segmented aperture.

    Parameters
    ----------
    side : float
        side length (and/or radius) of hexagon, in meters.
    rings : integer
        The number of rings of hexagons to include, not counting the central segment
        (i.e. 2 for a JWST-like aperture, 3 for a Keck-like aperture, and so on)
    gap : float
        Gap between adjacent segments, in meters.
    """

    side: float
    rings: int
    gap: float


@dataclass
class SecondaryObscuration:
    """Parameters to define the central obscuration of an on-axis telescope.

    The parameters include secondary mirror and supports.

    Parameters
    ----------
    secondary_radius : float
        Radius of the circular secondary obscuration, in meters.
    n_supports : int
        Number of secondary mirror supports ("spiders"). These will be
        spaced equally around a circle.
    support_width : float
        Width of each support, in meters.
    """

    secondary_radius: float
    n_supports: int
    support_width: float


@dataclass
class ZernikeWFE:
    """Parameters to define an optical element in terms of its Zernike components.

    Parameters
    ----------
    radius : float
        Pupil radius, in meters, over which the Zernike terms should be
        computed such that rho = 1 at r = `radius`.
    coefficients : iterable of floats
        Specifies the coefficients for the Zernike terms, ordered
        according to the convention of Noll et al. JOSA 1976. The
        coefficient is in meters of optical path difference (not waves).
    aperture_stop : float
    """

    radius: float
    coefficients: Sequence[float]
    aperture_stop: float


@dataclass
class SineWaveWFE:
    """Parameters to define a single sine wave ripple across the optic.

    Parameters
    ----------
    spatialfreq : float
    amplitude : float
    rotation : float
    """

    spatialfreq: float
    amplitude: float
    rotation: float


# Define a type alias
OpticalParameter = Union[
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
NewOpticalParameter = Union[
    CircularAperture,
    NewThinLens,
    SquareAperture,
    RectangleAperture,
    HexagonAperture,
    MultiHexagonalAperture,
    SecondaryObscuration,
    ZernikeWFE,
    SineWaveWFE,
]


def create_optical_parameter(dct: Mapping) -> OpticalParameter:
    """Create a new ``OpticalParameter`` based on a dictionary.

    Parameters
    ----------
    dct : dict
        Dictionary to convert

    Returns
    -------
    OpticalParameter
        New parameters.
    """
    if dct["item"] == "CircularAperture":
        return CircularAperture(radius=dct["radius"])

    elif dct["item"] == "ThinLens":
        return ThinLens(
            nwaves=dct["nwaves"],
            radius=dct["radius"],
        )

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


def new_create_optical_parameter(dct: Mapping) -> NewOpticalParameter:
    """Create a new ``OpticalParameter`` based on a dictionary.

    Parameters
    ----------
    dct : dict
        Dictionary to convert

    Returns
    -------
    OpticalParameter
        New parameters.
    """
    if dct["item"] == "CircularAperture":
        return CircularAperture(radius=dct["radius"])

    elif dct["item"] == "ThinLens":
        return NewThinLens(
            nwaves=dct["nwaves"],
            radius=dct["radius"],
            reference_wavelength=dct["reference_wavelength"],
        )

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
    optical_system: Sequence[Mapping],
) -> Sequence[OpticalParameter]:
    """Create a list of ``OpticalParameters``.

    Parameters
    ----------
    optical_system : ``list`` of ``dict``
        List to convert.

    Returns
    -------
    ``list`` of ``OpticalParameter``
        A new list of parameters.
    """
    return [create_optical_parameter(dct) for dct in optical_system]


def create_optical_item(
    param: OpticalParameter,
    wavelength: float,
) -> "op.OpticalElement":
    """Create a new poppy ``OpticalElement``.

    Parameters
    ----------
    param : ``OpticalParameter``
        Pyxel Optical parameters to create a poppy ``OpticalElement``.
    wavelength : float

    Returns
    -------
    ``OpticalElement``
        A new poppy ``OpticalElement``.
    """
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


def new_create_optical_item(param: NewOpticalParameter) -> "op.OpticalElement":
    """Create a new poppy ``OpticalElement``.

    Parameters
    ----------
    param : ``NewOpticalParameter``
        Pyxel Optical parameters to create a poppy ``OpticalElement``.

    Returns
    -------
    ``OpticalElement``
        A new poppy ``OpticalElement``.
    """
    if isinstance(param, CircularAperture):
        return op.CircularAperture(radius=param.radius)

    elif isinstance(param, NewThinLens):
        return op.ThinLens(
            nwaves=param.nwaves,
            reference_wavelength=param.reference_wavelength,
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
    wavelengths: Sequence[float],
    fov_arcsec: float,
    pixelscale: float,
    optical_elements: Sequence["op.OpticalElement"],
) -> tuple[fits.PrimaryHDU, fits.PrimaryHDU]:
    """Calculate the point spread function for the given optical system.

    Parameters
    ----------
    wavelengths : sequence of float
        Wavelengths of incoming light in meters.
    fov_arcsec : float, optional
        Field Of View on detector plane in arcsec.
    pixelscale : float
        Pixel scale on detector plane (arcsec/pixel).
        Defines sampling resolution of :term:`PSF`.
    optical_elements : list of OpticalElement
        List of optical elements to apply.

    Returns
    -------
    Tuple of two :term:`FITS`
        Tuple of psf and intermediate wavefronts.
    """
    if not WITH_POPPY:
        raise ModuleNotFoundError(
            "Missing optional package 'poppy'.\n"
            "Please install it with 'pip install pyxel-sim[model]' "
            "or 'pip install pyxel-sim[all]'"
        )

        # Calculate a monochromatic PSF
    output_fits: Sequence[fits.hdu.image.PrimaryHDU]
    wavefronts: Sequence[op.Wavefront]

    # Create Instrument
    instrument = op.Instrument(name="inst")

    instrument.pixelscale = pixelscale

    instrument.optsys = op.OpticalSystem(npix=1000)

    # Create the optical element(s)
    # osys = op.OpticalSystem(npix=1000)  # default: 1024

    element: "op.OpticalElement"
    for element in optical_elements:
        instrument.optsys.add_pupil(element)

    instrument.optsys.add_detector(
        pixelscale=pixelscale,
        fov_arcsec=fov_arcsec,
    )

    output_fits, wavefronts = instrument.calc_datacube(wavelengths=wavelengths)

    return output_fits, wavefronts


def apply_convolution(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve an array in 2D or 3D.

    Parameters
    ----------
    data : ndarray
        2D or 3D Array to be convolved with kernel_2d.
    kernel : ndarray
        The convolution kernel.

    Returns
    -------
    ndarray
        A convolved array.
    """
    mean = np.mean(data)

    array = convolve_fft(
        data,
        kernel=kernel,
        boundary="fill",
        fill_value=mean,
    )

    return array


def optical_psf(
    detector: Detector,
    wavelength: float,
    fov_arcsec: float,
    optical_system: Sequence[Mapping[str, Any]],
) -> None:
    """Model function for poppy optics model: convolve photon array with psf.

    Parameters
    ----------
    detector : Detector
        Pyxel Detector object.
    wavelength : float
        Wavelength of incoming light in meters.
    fov_arcsec : float
        Field Of View on detector plane in arcsec.
    optical_system : list of dict
        List of optical elements before detector with their specific arguments.
    """
    logging.getLogger("poppy").setLevel(
        logging.WARNING
    )  # TODO: Fix this. See issue #81

    # Validation and Conversion stage
    # These steps will be probably moved into the YAML engine
    if wavelength < 0.0 or fov_arcsec < 0.0 or detector.geometry.pixel_scale < 0.0:
        raise ValueError(
            "Expecting strictly positive value for 'wavelength', "
            "'fov_arcsec' and 'pixel_scale'."
        )

    # Convert 'optical_system' to 'optical_parameters'
    optical_parameters: Sequence[OpticalParameter] = [
        create_optical_parameter(dct) for dct in optical_system
    ]

    optical_elements: Sequence["op.OpticalElement"] = [
        create_optical_item(param=param, wavelength=wavelength)
        for param in optical_parameters
    ]

    # Processing
    # Get a Point Spread Function
    image_3d: fits.PrimaryHDU
    wavefront_3d: fits.PrimaryHDU
    image_3d, wavefront_3d = calc_psf(
        wavelengths=[wavelength],
        fov_arcsec=fov_arcsec,
        pixelscale=detector.geometry.pixel_scale,
        optical_elements=optical_elements,
    )

    data_2d: np.ndarray = image_3d.data[0]

    # Convolution
    new_array_2d: np.ndarray = apply_convolution(
        data=detector.photon.array,
        kernel=data_2d,
    )

    detector.photon.array = new_array_2d


def optical_psf_multi_wavelength(
    detector: Detector,
    wavelengths: tuple[float, float],
    fov_arcsec: float,
    optical_system: Sequence[Mapping[str, Any]],
) -> None:
    """Model function for poppy optics model: convolve photon array with psf.

    Parameters
    ----------
    detector : Detector
        Pyxel Detector object.
    wavelengths : tuple of floats
        Wavelengths of incoming light in meters.
    fov_arcsec : float
        Field Of View on detector plane in arcsec.
    optical_system : list of dict
        List of optical elements before detector with their specific arguments.
    """
    import xarray as xr
    from astropy.units import Quantity

    logging.getLogger("poppy").setLevel(
        logging.WARNING
    )  # TODO: Fix this. See issue #81

    # Validation and Conversion stage
    # These steps will be probably moved into the YAML engine
    if len(wavelengths) != 2:
        raise ValueError("Expecting two wavelengths in parameter 'wavelengths'.")

    min_wavelength, max_wavelength = wavelengths
    if min_wavelength <= 0:
        raise ValueError("Expecting strictly positive value for the 'wavelengths'")

    if min_wavelength > max_wavelength:
        raise ValueError(
            f"Min wavelength must be smaller that max wavelength. Got: {wavelengths!r}"
        )

    if fov_arcsec < 0.0 or detector.geometry.pixel_scale < 0.0:
        raise ValueError(
            "Expecting strictly positive value for " "'fov_arcsec' and 'pixel_scale'."
        )

    # Convert 'optical_system' to 'optical_parameters'
    optical_parameters: Sequence[NewOpticalParameter] = [
        new_create_optical_parameter(dct) for dct in optical_system
    ]

    optical_elements: Sequence["op.OpticalElement"] = [
        new_create_optical_item(param=param) for param in optical_parameters
    ]

    # Get current wavelengths (in nm)
    start_wavelength = Quantity(min_wavelength, unit="m")
    end_wavelength = Quantity(max_wavelength, unit="m")
    wavelengths_nm: Quantity = Quantity(
        detector.photon.array_3d["wavelength"], unit="nm"
    )

    selected_wavelengths_nm: Quantity = wavelengths_nm[
        np.logical_and(
            wavelengths_nm >= start_wavelength, wavelengths_nm <= end_wavelength
        )
    ]
    if selected_wavelengths_nm.size == 0:
        raise ValueError

    # Processing
    # Get a Point Spread Function
    image_3d: fits.PrimaryHDU
    wavefront_3d: fits.PrimaryHDU
    image_3d, wavefront_3d = calc_psf(
        wavelengths=selected_wavelengths_nm.to("m").value,
        fov_arcsec=fov_arcsec,
        pixelscale=detector.geometry.pixel_scale,
        optical_elements=optical_elements,
    )

    # Convolution
    new_array_3d: np.ndarray = apply_convolution(
        data=detector.photon.array_3d.to_numpy(),
        kernel=image_3d.data,
    )

    array_3d = xr.DataArray(
        new_array_3d,
        dims=["wavelength", "y", "x"],
        coords={"wavelength": selected_wavelengths_nm.value},
    )
    detector.photon.array_3d = array_3d

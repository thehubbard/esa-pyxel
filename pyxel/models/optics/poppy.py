#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Pyxel photon generator models."""
import logging
import typing as t

import numpy as np
import poppy as op
from matplotlib import pyplot as plt
from scipy import signal

from pyxel.data_structure import Photon
from pyxel.detectors import Detector


# @validators.validate
# @config.argument(name='image_file', label='', validate=check_path)
def optical_psf(
    detector: Detector,
    wavelength: float,
    pixelscale: float,
    fov_pixels: int,
    optical_system: list,
    fov_arcsec: t.Optional[float] = None,
) -> None:
    """POPPY (Physical Optics Propagation in PYthon) model wrapper.

    It calculates the optical Point Spread Function of an optical system.

    Documentation:
    https://poppy-optics.readthedocs.io/en/stable/index.html

    detector: Detector
        Pyxel Detector object.
    wavelength: float
        Wavelength of incoming light in meters.
    pixelscale: float
        Pixel scale on detector plane (micron/pixel or arcsec/pixel).
        Defines sampling resolution of PSF.
    fov_pixels: int
        Field Of View on detector plane in pixel.
    optical_system:
        List of optical elements before detector with their specific arguments.

        See details about POPPY Optical Element classes:
        https://poppy-optics.readthedocs.io/en/stable/available_optics.html

        Supported optical elements:

        - ``CircularAperture``
        - ``SquareAperture``
        - ``RectangularAperture``
        - ``HexagonAperture``
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

    fov_arcsec: float, optional
        Field Of View on detector plane in arcsec.
    """
    logging.getLogger("poppy").setLevel(logging.WARNING)

    if fov_arcsec:  # TODO
        raise NotImplementedError

    osys = op.OpticalSystem(npix=1000)  # default: 1024

    for item in optical_system:
        if item["item"] == "CircularAperture":
            optical_item = op.CircularAperture(radius=item["radius"])
        elif item["item"] == "ThinLens":
            optical_item = op.ThinLens(
                nwaves=item["nwaves"],
                reference_wavelength=wavelength,
                radius=item["radius"],
            )
        elif item["item"] == "SquareAperture":
            optical_item = op.SquareAperture(size=item["size"])
        elif item["item"] == "RectangularAperture":
            optical_item = op.RectangleAperture(
                width=item["width"], height=item["height"]
            )  # m
        elif item["item"] == "HexagonAperture":
            optical_item = op.HexagonAperture(side=item["side"])
        elif item["item"] == "SecondaryObscuration":
            optical_item = op.SecondaryObscuration(
                secondary_radius=item["secondary_radius"],
                n_supports=item["n_supports"],
                support_width=item["support_width"],
            )  # cm
        elif item["item"] == "ZernikeWFE":
            optical_item = op.ZernikeWFE(
                radius=item["radius"],
                coefficients=item["coefficients"],  # list of floats
                aperture_stop=item["aperture_stop"],
            )  # bool
        elif item["item"] == "SineWaveWFE":
            optical_item = op.SineWaveWFE(
                spatialfreq=item["spatialfreq"],  # 1/m
                amplitude=item["amplitude"],  # um
                rotation=item["rotation"],
            )
        else:
            raise NotImplementedError
        osys.add_pupil(optical_item)

    osys.add_detector(
        pixelscale=pixelscale,
        # fov_arcsec=None,
        fov_pixels=fov_pixels,
    )

    psf = osys.calc_psf(
        wavelength=wavelength,
        return_intermediates=True,
        # return_final=True,
        display_intermediates=True,
        normalize="last",
    )  # TODO NORMALIZATION!!!!

    # psf[0][0].data == psf[1][-1].intensity

    # plt.figure()
    # ax_orig = plt.gca()
    # ax_orig.imshow(detector.photon.array, cmap='gray')
    # ax_orig.set_title('Original')
    # ax_orig.set_axis_off()

    # Convolution
    a, b = detector.photon.array.shape
    new_shape = (a + 2 * fov_pixels, b + 2 * fov_pixels)
    array = np.zeros(new_shape, detector.photon.array.dtype)
    roi = slice(fov_pixels, fov_pixels + a), slice(fov_pixels, fov_pixels + b)
    array[roi] = detector.photon.array

    array = signal.convolve2d(
        array, psf[0][0].data, mode="same", boundary="fill", fillvalue=0
    )

    detector.photon = Photon(array)

    # plt.figure()
    # ax_int = plt.gca()
    # ax_int.imshow(array, cmap='gray')
    # ax_int.set_title('Convolution with intensity')
    # ax_int.set_axis_off()

    # plt.show()

    # conv_with_wavefront = signal.convolve2d(detector.photon.array, psf[1][-1].wavefront,
    #                                         mode='same', boundary='fill', fillvalue=0)

    plt.close("all")  # TODO

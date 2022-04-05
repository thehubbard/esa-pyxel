#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
from astropy.convolution import convolve_fft
import numpy as np
import typing as t
from pyxel.inputs import load_image

def apply_psf(array: np.ndarray, psf: np.ndarray, normalize_kernel: bool = True) -> np.ndarray:
    """

    Parameters
    ----------
    array: ndarray
    psf: ndarray
    normalize_kernel: bool

    Returns
    -------

    """

    mean = np.mean(data_2d)

    array_2d = convolve_fft(
        array,
        kernel=psf,
        boundary="fill",
        fill_value=mean,
        crop=False,
        normalize_kernel = normalize_kernel,
    )

    return array_2d

def psf(detector: Detector, filename: t.Union[str, Path], normalize_kernel: bool = True) -> None:
    """Load a PSF from file and apply convolution with PSF to the photon array.

    Parameters
    ----------
    detector: Detector
    filename: Path or str
    normalize_kernel: bool
    """
    psf = load_image(filename)

    detector.photon.array = apply_psf(array=detector.photon.array, psf=psf, normalize_kernel=normalize_kernel)
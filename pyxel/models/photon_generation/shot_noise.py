#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Pyxel photon generator models: photon shot noise."""
import typing as t
from typing.extensions import Literal

import numpy as np

from pyxel.detectors import Detector
from pyxel.util import temporary_random_state


def compute_poisson_noise(array: np.ndarray):
    """Compute Poisson noise using the input array.

    Parameters
    ----------
    array: np.ndarray
        Input array.

    Returns
    -------
    output np.ndarray
        Output array.
    """
    output = np.random.poisson(lam=array).astype(np.float64)
    return output

def compute_gaussian_noise(array: np.ndarray):
    """Compute Gaussian noise using the input array. Standard deviation is square root of the values.

    Parameters
    ----------
    array: np.ndarray
        Input array.

    Returns
    -------
    output np.ndarray
        Output array.
    """
    output = np.random.normal(loc=array, scale=np.sqrt(array))
    return output

def compute_noise(array: np.ndarray, type: t.Literal["possion", "normal"] = "poisson"):
    """Compute shot noise for an input array. It can be either Poisson noise or Gaussian.

    Parameters
    ----------
    array: np.ndarray
        Input array.
    type: str
        Choose either 'poisson' or 'normal'. Default is Poisson noise.

    Returns
    -------
    output np.ndarray
        Output array.
    """
    if type == "poisson":
        output = compute_poisson_noise(array)
    elif type == "normal":
        output = compute_gaussian_noise(array)
    else:
        raise ValueError("Invalid noise type!")


@temporary_random_state
def shot_noise(detector: Detector, type: t.Literal["possion", "normal"] = "poisson", seed: t.Optional[int] = None) -> None:
    """Add shot noise to the flux of photon per pixel. It can be either Poisson noise or Gaussian.

    Parameters
    ----------
    detector: Detector
        Pyxel Detecotr object.
    type: str
        Choose either 'poisson' or 'normal'. Default is Poisson noise.
    seed: int, optional
        Random seed.
    """
    noise_array = compute_noise(array=detector.photon.array, type=type)
    detector.photon.array = noise_array

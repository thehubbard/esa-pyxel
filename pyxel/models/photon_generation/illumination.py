#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Pyxel photon generator models."""
import typing as t
from typing_extensions import Literal

import numpy as np

from pyxel.detectors import Detector


def rectangular_hole(
    shape: t.Tuple[int, int],
    level: float,
    hole_size: t.Optional[t.Sequence[int]] = None,
    hole_center: t.Optional[t.Sequence[int]] = None,
) -> np.ndarray:
    """Calculate an image of a rectangular hole.

    Parameters
    ----------
    shape: tuple
        Shape of the output array.
    level: float
        Flux of photon per pixel.
    hole_size: list or tuple, optional
        List or tuple of length 2, integers defining the diameters of the rectangular hole
        in vertical and horizontal directions.
    hole_center: list or tuple, optional
        List or tuple of length 2, two integers (row and column number),
        defining the coordinates of the center of the rectangular hole.

    Returns
    -------
    photon_array: np.ndarray
        Output numpy array.
    """
    if not hole_size:
        raise ValueError("hole_size argument should be defined for illumination model")
    if hole_size and not len(hole_size) == 2:
        raise ValueError("Hole size should be a sequence of length 2!")
    if hole_center and not len(hole_center) == 2:
        raise ValueError("Hole size should be a sequence of length 2!")

    photon_array = np.zeros(shape, dtype=float)
    if hole_center is not None:
        if not (
            (0 <= hole_center[0] <= shape[0]) and (0 <= hole_center[1] <= shape[1])
        ):
            raise ValueError('Argument "hole_center" should be inside Photon array.')
    else:
        hole_center = [int(shape[0] / 2), int(shape[1] / 2)]
    p = hole_center[0] - int(hole_size[0] / 2)
    q = hole_center[1] - int(hole_size[1] / 2)
    p0 = int(np.clip(p, a_min=0, a_max=shape[0]))
    q0 = int(np.clip(q, a_min=0, a_max=shape[1]))
    photon_array[slice(p0, p + hole_size[0]), slice(q0, q + hole_size[1])] = level

    return photon_array


def elliptic_hole(
    shape: t.Tuple[int, int],
    level: float,
    hole_size: t.Optional[t.Sequence[int]] = None,
    hole_center: t.Optional[t.Sequence[int]] = None,
) -> np.ndarray:
    """Calculate an image of an elliptic hole.

    Parameters
    ----------
    shape: tuple
        Shape of the output array.
    level: float
        Flux of photon per pixel.
    hole_size: list or tuple, optional
        List or tuple of length 2, integers defining the diameters of the elliptic hole
        in vertical and horizontal directions.
    hole_center: list or tuple, optional
        List or tuple of length 2, two integers (row and column number),
        defining the coordinates of the center of the elliptic hole.

    Returns
    -------
    photon_array: np.ndarray
        Output numpy array.
    """
    if not hole_size:
        raise ValueError("hole_size argument should be defined for illumination model")
    if hole_size and not len(hole_size) == 2:
        raise ValueError("Hole size should be a sequence of length 2!")
    if hole_center and not len(hole_center) == 2:
        raise ValueError("Hole size should be a sequence of length 2!")

    photon_array = np.zeros(shape, dtype=float)
    if hole_center is not None:
        if not (
            (0 <= hole_center[0] <= shape[0]) and (0 <= hole_center[1] <= shape[1])
        ):
            raise ValueError('Argument "hole_center" should be inside Photon array.')
    else:
        hole_center = [int(shape[0] / 2), int(shape[1] / 2)]
    y, x = np.ogrid[: shape[0], : shape[1]]
    dist_from_center = np.sqrt(
        ((x - hole_center[1]) / float(hole_size[1]/2)) ** 2
        + ((y - hole_center[0]) / float(hole_size[0]/2)) ** 2
    )
    photon_array[dist_from_center < 1] = level
    return photon_array


def calculate_illumination(
    shape: t.Tuple[int, int],
    level: float,
    option: str = "uniform",
    hole_size: t.Optional[t.Sequence[int]] = None,
    hole_center: t.Optional[t.Sequence[int]] = None,
) -> np.ndarray:
    """Calculate the array of photons uniformly over the entire array or over a hole.

    Parameters
    ----------
    shape: tuple
        Shape of the output array.
    level: float
        Flux of photon per pixel.
    option: str{'uniform', 'elliptic_hole', 'rectangular_hole'}
        A string indicating the type of illumination:
        - ``uniform``
           Uniformly fill the entire array with photon. (Default)
        - ``elliptic_hole``
           Mask with elliptic hole.
        - ``rectangular_hole``
           Mask with rectangular hole.
    hole_size: list or tuple, optional
        List or tuple of length 2, integers defining the diameters of the elliptic or rectangular hole
        in vertical and horizontal directions.
    hole_center: list or tuple, optional
        List or tuple of length 2, two integers (row and column number),
        defining the coordinates of the center of the elliptic or rectangular hole.

    Returns
    -------
    photon_array: np.ndarray
        Output numpy array.
    """
    if option == "uniform":
        photon_array = np.ones(shape, dtype=float) * level
    elif option == "rectangular_hole":
        photon_array = rectangular_hole(
            shape=shape, hole_size=hole_size, hole_center=hole_center, level=level
        )
    elif option == "elliptic_hole":
        photon_array = elliptic_hole(
            shape=shape, hole_size=hole_size, hole_center=hole_center, level=level
        )
    else:
        raise NotImplementedError

    return photon_array


def illumination(
    detector: Detector,
    level: float,
    option: Literal["uniform", "rectangular_hole", "elliptic_hole"] = "uniform",
    hole_size: t.Optional[t.Sequence[int]] = None,
    hole_center: t.Optional[t.Sequence[int]] = None,
    time_scale: float = 1.0,
) -> None:
    """Generate photon uniformly over the entire array or over a hole.

    detector: Detector
        Pyxel Detector object.
    level: float
        Flux of photon per pixel.
    option: str{'uniform', 'elliptic_hole', 'rectangular_hole'}
        A string indicating the type of illumination:
        - ``uniform``
           Uniformly fill the entire array with photon. (Default)
        - ``elliptic_hole``
           Mask with elliptic hole.
        - ``rectangular_hole``
           Mask with rectangular hole.
    hole_size: list or tuple, optional
        List or tuple of length 2, integers defining the diameters of the elliptic or rectangular hole
        in vertical and horizontal directions.
    hole_center: list or tuple, optional
        List or tuple of length 2, two integers (row and column number),
        defining the coordinates of the center of the elliptic or rectangular hole.
    time_scale: float
        Time scale of the photon flux, default is 1 second. 0.001 would be ms.
    """
    shape = (detector.geometry.row, detector.geometry.col)

    photon_array = calculate_illumination(
        shape=shape,
        level=level,
        option=option,
        hole_size=hole_size,
        hole_center=hole_center,
    )

    photon_array = photon_array * (detector.time_step / time_scale)

    try:
        detector.photon.array += photon_array
    except ValueError as ex:
        raise ValueError("Shapes of arrays do not match") from ex

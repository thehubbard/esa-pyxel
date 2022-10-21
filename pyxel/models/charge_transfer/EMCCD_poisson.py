#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Model for replicating the gain register in an EMCCD"""

import numpy as np
from pyxel.detectors import CCD
import numba


def multiplication_register(
        detector: CCD,
        total_gain: int,
        gain_elements: int,
) -> None:
    detector.pixel.array = multiplication_register_poisson(
        detector.pixel.array,
        total_gain,
        gain_elements).astype(np.float)

@numba.njit
def poisson_register(lam,
                     image_cube_pix,
                     new_image_cube_pix,
                     gain_elements):
    x=0
    while x != gain_elements:
        if x == 0:

            electron_gain = np.random.poisson(lam, int(image_cube_pix))

            new_image_cube_pix = np.round(
                image_cube_pix + np.sum((electron_gain), 0
            ))

        else:  # subsequent elements continue adding to the counter instead

            electron_gain = np.random.poisson(lam, int(new_image_cube_pix))

            new_image_cube_pix = np.round(
                new_image_cube_pix + np.sum(electron_gain), 0
            )
        x += 1
    return new_image_cube_pix


@numba.njit
def multiplication_register_poisson(
        image_cube: np.ndarray,
        total_gain: int,
        gain_elements: int,
) -> None:

    new_image_cube = np.zeros_like(image_cube)
    new_image_cube = new_image_cube.astype(np.int32)

    lam = total_gain ** (1 / gain_elements) - 1

    for j in range(0, image_cube.shape[0]):
        for i in range(0, image_cube.shape[1]):

            if image_cube[j, i] < 0:
                new_image_cube[j, i] = poisson_register(lam, 0, new_image_cube[j, i], gain_elements)
            else:
                new_image_cube[j, i] = poisson_register(lam, image_cube[j, i], new_image_cube[j, i], gain_elements)

    return new_image_cube


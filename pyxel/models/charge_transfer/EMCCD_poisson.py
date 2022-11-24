#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Model for replicating the gain register in an EMCCD"""

import numba
import numpy as np

from pyxel.detectors import CCD

"""Main multiplication register that takes in CCD detector along with the gain and the total elements of the EMCCD 
multiplication register """


def multiplication_register(
        detector: CCD,
        total_gain: int,
        gain_elements: int,
) -> None:
    if total_gain < 0 or gain_elements < 0:
        raise ValueError("Wrong input parameter")

    detector.pixel.array = multiplication_register_poisson(
        image_cube=detector.pixel.array,
        total_gain=total_gain,
        gain_elements=gain_elements,
    ).astype(float)


"""This function cycles through each pixel within the image provided, calculates the signal gain from the 
multiplication register and once all pixels have been cycles through returns a final image with signal added """


@numba.njit
def poisson_register(lam, image_cube_pix, gain_elements):
    new_image_cube_pix = image_cube_pix

    for _ in range(gain_elements):
        electron_gain = np.random.poisson(lam, size=int(new_image_cube_pix))
        new_image_cube_pix = np.round(image_cube_pix + np.sum(electron_gain))

    return new_image_cube_pix


"""The function that does the signal gain calculation through the use of poissonian statistics. A single pixel is 
inputted and is iterated through the total number of gain elements provided with the result being the resultant signal 
from the pixel going through the multiplication process """


@numba.njit
def multiplication_register_poisson(
        image_cube: np.ndarray,
        total_gain: int,
        gain_elements: int,
) -> np.ndarray:
    new_image_cube = np.zeros_like(image_cube, dtype=np.int32)

    lam = total_gain ** (1 / gain_elements) - 1
    yshape, xshape = image_cube.shape

    for j in range(0, yshape):
        for i in range(0, xshape):

            if image_cube[j, i] < 0:
                new_image_cube[j, i] = poisson_register(
                    lam=lam, image_cube_pix=0, gain_elements=gain_elements
                )
            else:
                new_image_cube[j, i] = poisson_register(
                    lam=lam,
                    image_cube_pix=image_cube[j, i],
                    gain_elements=gain_elements,
                )

    return new_image_cube

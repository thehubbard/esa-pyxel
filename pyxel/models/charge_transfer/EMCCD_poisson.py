#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Model for replicating the gain register in an EMCCD"""

import numpy as np
from collections import Counter
from pyxel.detectors import CCD


def multiplication_register_poisson(
        detector: CCD,
        total_gain: int,
        gain_elements: int,
) -> None:

    image_cube = np.asarray(detector.pixel.array, dtype=float)
    image_cube = image_cube
    new_image_cube = np.zeros_like(image_cube)
    new_image_cube = new_image_cube.astype(np.int32)

    lam = total_gain ** (1 / gain_elements) - 1

    for j in range(0, image_cube.shape[0]):
        for i in range(0, image_cube.shape[1]):
            x = 0

            while x != gain_elements:
                if x == 0:
                    electron_gain = np.random.poisson(lam, int(image_cube[j, i]))
                    electron_counter = Counter(electron_gain)  # Counts how many electrons have been gained
                    new_image_cube[j, i] = np.round(image_cube[j, i] + np.sum(electron_gain), 0)

            else:  # subsequent elements continue adding to the counter instead

                electron_gain = np.random.poisson(lam, new_image_cube[j, i])
                electron_counter = Counter(electron_gain) + Counter(electron_counter)
                new_image_cube[j, i] = np.round(new_image_cube[j, i] + np.sum(electron_gain), 0)
                x = x + 1

    detector.pixel.array = new_image_cube

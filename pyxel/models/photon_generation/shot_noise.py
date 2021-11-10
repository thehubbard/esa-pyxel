#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Pyxel photon generator models: photon shot noise."""
import typing as t

import numpy as np

from pyxel.detectors import Detector
from pyxel.util import temporary_random_state

# TODO: random, docstring, private function
# TODO: normal and poisson versions, maybe as an argument,


# TODO: Fix this
# @validators.validate
# @config.argument(name='seed', label='random seed', units='', validate=checkers.check_type(int))
@temporary_random_state
def shot_noise(detector: Detector, seed: t.Optional[int] = None) -> None:
    """Add shot noise to the number of photon per pixel.

    Parameters
    ----------
    detector: Detector
        Pyxel Detecotr object.
    seed: int, optional
        Random seed.
    """
    detector.photon.array = np.random.poisson(lam=detector.photon.array)

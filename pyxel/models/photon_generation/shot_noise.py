"""Pyxel photon generator models: photon shot noise."""
import logging
import numpy as np
from pyxel.detectors import Detector
import typing as t
from esapy_config import funcargs, checkers


# FRED: Remove the following decorators
@funcargs.validate
@funcargs.argument(name='seed', label='random seed', units='', validate=checkers.check_type_function(int))
def shot_noise(detector: Detector, random_seed: t.Optional[int] = None) -> None:
    """Add shot noise to the number of photon per pixel.

    :param detector: Pyxel Detector object
    :param random_seed: int seed
    """
    log = logging.getLogger('pyxel')
    log.info('')
    if random_seed:
        np.random.seed(random_seed)
    detector.photon.array = np.random.poisson(lam=detector.photon.array)

"""Pyxel photon generator models."""
import logging
import numpy as np
import pyxel
from pyxel import check_type
from pyxel.detectors.detector import Detector


@pyxel.validate
@pyxel.argument(name='seed', label='random seed', units='', validate=check_type(int))
def shot_noise(detector: Detector, random_seed: int = None):
    """Add shot noise to the number of photons.

    :param detector: Pyxel Detector object
    :param random_seed: int seed
    """
    logger = logging.getLogger('pyxel')
    logger.info('')
    if random_seed:
        np.random.seed(random_seed)
    detector.photons.array = np.random.poisson(lam=detector.photons.array)

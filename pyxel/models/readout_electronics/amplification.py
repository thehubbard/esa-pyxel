#   --------------------------------------------------------------------------
#   Copyright 2018 SCI-FIV, ESA (European Space Agency)
#   --------------------------------------------------------------------------
"""Readout electronics model."""
import logging
# import pyxel
from pyxel.detectors.detector import Detector
# from astropy import units as u


# @pyxel.validate
# @pyxel.argument(name='', label='', units='', validate=)
def simple_amplifier(detector: Detector):
    """Amplify signal.

    amp - Gain of output amplifier, a1 - Gain of the signal processor

    :param detector: Pyxel Detector object
    """
    logger = logging.getLogger('pyxel')
    logger.info('')
    char = detector.characteristics
    detector.signal.array *= char.amp * char.a1
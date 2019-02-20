#   --------------------------------------------------------------------------
#   Copyright 2018 SCI-FIV, ESA (European Space Agency)
#   --------------------------------------------------------------------------
"""Charge readout model."""
import logging
# import pyxel
from pyxel.detectors.detector import Detector
# from astropy import units as u


# @pyxel.validate
# @pyxel.argument(name='', label='', units='', validate=)
def simple_measurement(detector: Detector):
    """Create signal array from pixel array.

    detector Signal unit: Volt

    :param detector: Pyxel Detector object
    """
    logger = logging.getLogger('pyxel')
    logger.info('')
    char = detector.characteristics

    array = detector.pixels.array * char.sv
    detector.signal.array = array.astype('float64')
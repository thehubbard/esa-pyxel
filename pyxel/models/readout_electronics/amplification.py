#   --------------------------------------------------------------------------
#   Copyright 2019 SCI-FIV, ESA (European Space Agency)
#   --------------------------------------------------------------------------
"""Readout electronics model."""
import logging
from pyxel.detectors.detector import Detector
# from astropy import units as u


# @pyxel.validate
# @pyxel.argument(name='', label='', units='', validate=)
def simple_amplifier(detector: Detector) -> None:
    """Amplify signal.

    amp - Gain of output amplifier, a1 - Gain of the signal processor

    :param detector: Pyxel Detector object
    """
    logging.info('')
    char = detector.characteristics
    detector.signal.array *= char.amp * char.a1

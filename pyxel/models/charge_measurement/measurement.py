#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Charge readout model."""
import logging

from pyxel.detectors import Detector


# TODO: keep name, private functions, more documentation, tests
def simple_measurement(detector: Detector) -> None:
    """Create signal array from pixel array.

    Parameters
    ----------
    detector
    gain
    """
    char = detector.characteristics

    array = detector.pixel.array * char.sv
    detector.signal.array = array.astype("float64")


# TODO: how is it different? rename or merge both into one model, get rid of gain
def simple_test(detector: Detector, gain: int) -> None:
    """Create signal array from pixel array.

    detector Signal unit: Volt

    :param detector: Pyxel Detector object
    """
    logging.info("")
    char = detector.characteristics

    array = detector.pixel.array * char.sv * gain
    detector.signal.array = array.astype("float64")

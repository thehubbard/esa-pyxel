#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Readout electronics model."""

from pyxel.detectors import MKID, Detector


# TODO: pure functions, documentation, copies!
def basic_processing(detector: Detector) -> None:
    """Create an image array from signal array.

    :param detector: Pyxel Detector object
    """

    detector.image.array = detector.signal.array


def phase_conversion(detector: MKID) -> None:
    """Create an image array from phase array.

    Parameters
    ----------
    detector: MKID
        Pyxel MKID detector object.
    """
    if not isinstance(detector, MKID):
        raise TypeError("Expecting a MKID object for the detector.")

    detector.image.array = detector.phase.array

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Non-Destructive Readout modes for CMOS-based detectors."""
from typing_extensions import Literal

from pyxel.detectors import CMOS


# TODO: find a way to integrate this in readout definition in running mode and remove this model for 1.0
# TODO: different readout noise for different readout modes?
def non_destructive_readout(
    detector: CMOS,
    mode: Literal["uncorrelated", "CDS", "Fowler-N", "UTR"],
    fowler_samples: int = 1,
) -> None:
    """Non-Destructive Readout modes for CMOS-based detectors.

    Parameters
    ----------
    detector : CMOS
        CMOS detector object.
    mode : str
        Valid values: 'uncorrelated', 'CDS', 'Fowler-N', 'UTR'
    fowler_samples : int

    Raises
    ------
    ValueError
        If the 'detector' is not properly initialized.
    NotImplementedError
        If 'mode' is not recognized.
    """
    if not detector.non_destructive_readout or not detector.is_dynamic:
        raise ValueError()

    if not detector.times_linear:
        raise ValueError()

    detector.read_out = False
    if mode == "uncorrelated":
        if detector.pipeline_count == detector.num_steps - 1:
            detector.read_out = True

    elif mode == "CDS":
        if detector.pipeline_count == 0 or detector.pipeline_count == (
            detector.num_steps - 1
        ):
            detector.read_out = True

    elif mode == "Fowler-N":
        nt = fowler_samples
        detector.read_out = True
        if nt <= detector.pipeline_count < (detector.num_steps - nt):
            detector.read_out = False

    elif mode == "UTR":
        detector.read_out = True

    else:
        raise NotImplementedError

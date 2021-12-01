#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""

import numpy as np

from pyxel.detectors import MKID


def apply_dead_time_filter(phase_2d: np.ndarray, maximum_count: float) -> np.ndarray:
    """Apply dead time filter.

    Parameters
    ----------
    phase_2d : ndarray
    maximum_count : float

    Returns
    -------
    ndarray
    """
    phase_clipped_2d = np.clip(phase_2d, a_min=None, a_max=maximum_count)

    return phase_clipped_2d


# TODO: more documentation (Enrico). See #324.
def dead_time_filter(detector: MKID, dead_time: float) -> None:
    """Dead time filter.

    This model is described in :cite:p:`2020:prodhomme` in section 3.3.1.

    Parameters
    ----------
    detector: MKID
        Pyxel Detector MKID object.
    dead_time : float
    """
    # Validation phase
    if not isinstance(detector, MKID):
        # Later, this will be checked in when YAML configuration file is parsed
        raise TypeError("Expecting a `MKID` object for 'detector'.")

    if dead_time <= 0.0:
        raise ValueError("'dead_time' must be strictly positive.")

    phase_2d = apply_dead_time_filter(
        phase_2d=detector.phase.array,
        maximum_count=1.0 / dead_time,
    )

    detector.phase.array = phase_2d

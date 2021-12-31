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

    The underlying physics of this model is described in :cite:p:`PhysRevB.104.L180506`; more information can be found on the website :cite:p:`Mazin`.

    Parameters
    ----------
    detector: MKID
        Pyxel Detector MKID object.
    dead_time : float
    """
    # Validation phase
    if not isinstance(detector, MKID):
        # Later, this will be checked in when YAML configuration file is parsed
        raise TypeError("Expecting an `MKID` object for 'detector'.")

    if dead_time <= 0.0:
        raise ValueError("'dead_time' must be strictly positive.")

    char = detector.characteristics

    k_B: float = 8.617333262145 * 1.e-5  # [eV K^-1]

    # Cf. "Strong Reduction of Quasiparticle Fluctuations in a Superconductor due to Decoupling of the Quasiparticle Number and Lifetime", De Rooij et al. (2021).

    Delta = 1.76 * k_B * char.T_c
    N_qp = 2. * char.V * char.N_0 * np.sqrt(2. * np.pi * k_B * char.T_op * Delta) * np.exp(- Delta / (k_B * char.T_op))
    R = char.tau_0 * char.N_0 * (k_B * char.T_c) ** 3 / (2. * Delta ** 2)
    
    tau_qp = char.V / (R * N_qp)
    # tau_apparent = 1. / (2. / (tau_qp * (1. + (char.tau_esc / char.tau_pb))))
    tau_apparent_sat = 1. / (2. / (tau_qp * (1. + (char.tau_esc / char.tau_pb))) + (1. / char.tau_sat))
    dead_time = tau_apparent_sat
    
    phase_2d = apply_dead_time_filter(
        phase_2d=detector.phase.array,
        maximum_count=1.0 / dead_time,
    )

    detector.phase.array = phase_2d

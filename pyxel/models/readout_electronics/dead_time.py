#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""

import astropy.constants as const
import astropy.units as u
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

    The underlying physics of this model is described in :cite:p:`PhysRevB.104.L180506`;
    more information can be found on the website :cite:p:`Mazin`.

    Parameters
    ----------
    detector: MKID
        Pyxel Detector MKID object
    dead_time : float

    Delta : float
        Superconducting gap energy
    N_qp : float
        Number of quasi-particles
    R : float
        Recombination constant

    tau_qp : float
        Intrinsic quasi-particle lifetime
    tau_apparent : float
        Apparent quasi-particle lifetime, without saturation lifetime
    tau_apparent_sat : float
        Apparent quasi-particle lifetime, including saturation lifetime
    """
    # Validation phase
    if not isinstance(detector, MKID):
        # Later, this will be checked in when YAML configuration file is parsed
        raise TypeError("Expecting an `MKID` object for 'detector'.")

    if dead_time <= 0.0:
        raise ValueError("'dead_time' must be strictly positive.")

    char = detector.characteristics

    # Boltzmann's constant [eV K^-1]
    boltzmann_cst: float = const.k_B.to(u.eV / u.K).value

    # Compute superconducting gap energy
    delta = 1.76 * boltzmann_cst * char.t_c

    # Compute number of quasiparticles in a superconducting volume V
    # TODO: check that T << (delta / boltzmann_cst)
    n_qp = (
        2.0
        * char.v
        * char.n_0
        * np.sqrt(2.0 * np.pi * boltzmann_cst * char.t_op * delta)
        * np.exp(-delta / (boltzmann_cst * char.t_op))
    )

    recombination_cst = (
        char.tau_0 * char.n_0 * (boltzmann_cst * char.t_c) ** 3 / (2.0 * delta ** 2)
    )

    # Compute intrinsic quasiparticle lifetime with respect to recombination
    tau_qp = char.v / (recombination_cst * n_qp)
    # tau_apparent = 1. / (2. / (tau_qp * (1. + (char.tau_esc / char.tau_pb))))
    tau_apparent_sat = 1.0 / (
        2.0 / (tau_qp * (1.0 + (char.tau_esc / char.tau_pb))) + (1.0 / char.tau_sat)
    )
    dead_time = tau_apparent_sat

    phase_2d = apply_dead_time_filter(
        phase_2d=detector.phase.array,
        maximum_count=1.0 / dead_time,
    )

    detector.phase.array = phase_2d

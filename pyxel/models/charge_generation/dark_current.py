#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Simple model to generate charges due to dark current process."""
import logging
import typing as t

import numpy as np

from pyxel.detectors import CMOS
from pyxel.util import temporary_random_state

# TODO: more documentation, refactoring, random, astropy constants


# TODO: Fix this
# @validators.validate
# @config.argument(name='detector', label='', units='', validate=checkers.check_type(CMOS))
@temporary_random_state
def dark_current_rule07(detector: CMOS, seed: t.Optional[int] = None) -> None:
    """Generate charge from dark current process.

    Parameters
    ----------
    detector: Detector
    seed: int, optional
    """
    # TODO: investigate on the knee of rule07 for higher 1/le*T values
    logging.info("")
    geo = detector.geometry
    temperature = detector.environment.temperature
    cutoff = detector.characteristics.cutoff

    amp_to_eps = 6.242e18  # conversion factor from Ampere to Electrons per second
    um2_to_cm2 = 1.0e-8
    conversion_factor = amp_to_eps * um2_to_cm2

    # pitch = 18              # um
    # Rule 07 empirical model parameters
    j0 = 8367.000019  # A/cm**2
    c = -1.162972237
    q = 1.602176624e-19  # Elementary charge (Coulomb)
    k = 1.38064852e-23  # Boltzmann constant (m2 kg s-2 K-1)

    def lambda_e(lambda_cutoff: float) -> float:
        """Compute lambda_e.

        :param lambda_cutoff: (int) Cut-off wavelength of the detector
        """
        lambda_scale = 0.200847413  # um
        lambda_threshold = 4.635136423  # um
        pwr = 0.544071282
        if lambda_cutoff < lambda_threshold:
            le = lambda_cutoff / (
                1
                - ((lambda_scale / lambda_cutoff) - (lambda_scale / lambda_threshold))
                ** pwr
            )
        else:
            le = lambda_cutoff
        return le

    # Rule07
    j = j0 * np.exp(c * (1.24 * q / k) * 1.0 / (lambda_e(cutoff) * temperature))
    dark = j * conversion_factor * geo.pixel_horz_size * geo.pixel_vert_size
    # The number of charge generated with Poisson distribution using rule07 empiric law for lambda
    charge_number = np.random.poisson(dark, size=(geo.row, geo.col))

    detector.charge.add_charge_array(charge_number)

#   --------------------------------------------------------------------------
#   Copyright 2019 SCI-FIV, ESA (European Space Agency)
#   Written by B. Serra
#   --------------------------------------------------------------------------
"""Pyxel persistence model."""
import logging

from pyxel.detectors import CMOS

import numpy as np


# @validators.validate
# @config.argument(name='', label='', units='', validate=)
def simple_persistence(
    detector: CMOS, trap_timeconstants: list, trap_densities: list
) -> None:
    """Trapping/detrapping charges."""
    logging.info("Persistence")
    if "persistence" not in detector._memory.keys():
        detector._memory["persistence"] = dict()
        for trap_density, trap_timeconstant in zip(trap_densities, trap_timeconstants):
            entry = "".join(
                ["trappedCharges_", str(trap_density), "-", str(trap_timeconstant)]
            )
            detector._memory["persistence"].update(
                {entry: np.zeros((detector.geometry.row, detector.geometry.col))}
            )
    else:
        for trap_density, trap_timeconstant in zip(trap_densities, trap_timeconstants):
            entry = "".join(
                ["trappedCharges_", str(trap_density), "-", str(trap_timeconstant)]
            )
            trapped_charges = detector._memory["persistence"][entry]
            # Trap density is a scalar for now, in the future we could feed maps?
            trapped_charges = trapped_charges + (0.5 / trap_timeconstant) * (
                detector.pixel.array * trap_density - trapped_charges
            )
            # Remove the trapped charges from the pixel
            detector.pixel.array -= trapped_charges.astype(np.int32)
            # Replace old trapped charges map in the detector's memory
            detector._memory["persistence"][entry] = trapped_charges

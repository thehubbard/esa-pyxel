#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""

import typing as t

import numpy as np

from pyxel.detectors import CCD

try:
    import arcticpy as ac

    WITH_ARTICPY = True
except ImportError:
    # No 'arcticpy' library
    WITH_ARTICPY = False


def arctic_add(
    detector: CCD,
    well_fill_power: float,
    trap_densities: list,
    trap_release_timescales: list,
    express: int = 0,
    # instant_traps: t.Sequence[t.Mapping[str, float]],
) -> None:
    """Add trap species.

    Parameters
    ----------
    detector : CCD
    well_fill_power
    trap_densities
    trap_release_timescales
    express
    """
    if not WITH_ARTICPY:
        raise RuntimeError(
            "ArCTIC python wrapper is not installed ! "
            "See https://github.com/jkeger/arctic"
        )

    ccd = ac.CCD(
        phases=[
            ac.CCDPhase(
                full_well_depth=detector.characteristics.fwc,
                well_fill_power=well_fill_power,
            )
        ]
    )

    roe = ac.ROE()

    traps = []  # type: t.List[ac.TrapInstantCapture]
    i = 0
    for trap_density in trap_densities:

        instant_traps = [
            {"density": trap_density, "release_timescale": trap_release_timescales[i]}
        ]  # type: t.Sequence[t.Mapping[str, float]]

        # Build the traps
        for trap_info in instant_traps:
            density = trap_info["density"]  # type: float
            release_timescale = trap_info["release_timescale"]  # type: float
            trap = ac.TrapInstantCapture(
                density=density, release_timescale=release_timescale
            )
            traps.append(trap)
        # Add CTI
        i += 1

    image_2d = np.asarray(detector.pixel.array, dtype=float)  # type: np.ndarray

    image_cti_added_2d = ac.add_cti(
        image=image_2d,
        parallel_traps=traps,
        parallel_ccd=ccd,
        parallel_roe=roe,
        parallel_express=express,
        verbosity=0,
    )

    detector.pixel.array = image_cti_added_2d


def arctic_remove(
    detector: CCD,
    well_fill_power: float,
    instant_traps: t.Sequence[t.Mapping[str, float]],
    num_iterations: int,
    express: int = 0,
) -> None:
    """Remove trap species.

    Parameters
    ----------
    detector
    well_fill_power
    instant_traps
    num_iterations
    express
    """
    if not WITH_ARTICPY:
        raise RuntimeError(
            "ArCTIC python wrapper is not installed ! "
            "See https://github.com/jkeger/arctic"
        )

    ccd = ac.CCD(
        well_fill_power=well_fill_power, full_well_depth=detector.characteristics.fwc
    )
    roe = ac.ROE()

    # Build the traps
    traps = []  # type: t.List[ac.Trap]
    for trap_info in instant_traps:
        density = trap_info["density"]  # type: float
        release_timescale = trap_info["release_timescale"]  # type: float

        trap = ac.Trap(density=density, release_timescale=release_timescale)
        traps.append(trap)

    # Remove CTI
    image_2d = np.asarray(detector.pixel.array, dtype=float)  # type: np.ndarray

    image_cti_removed = ac.remove_cti(
        image=image_2d,
        iterations=num_iterations,
        parallel_traps=traps,
        parallel_ccd=ccd,
        parallel_roe=roe,
        parallel_express=express,
    )

    detector.pixel.array = image_cti_removed

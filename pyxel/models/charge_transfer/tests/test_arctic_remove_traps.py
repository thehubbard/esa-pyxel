#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import numpy as np

from pyxel.detectors import CCD, CCDCharacteristics, CCDGeometry, Environment, Material
from pyxel.models.charge_transfer.arctic_cti.models_arctic_numba import (
    arctic_remove_numba,
)
from pyxel.models.charge_transfer.arctic_cti.models_arctic_vanilla import arctic_remove


def create_detector() -> CCD:
    """Create a valid `CCD` object."""
    pixel_2d = np.zeros((100, 10), dtype=int)

    pixel_2d[25:50, :] = 100
    pixel_2d[75:, :] = 100

    pixel_2d = pixel_2d[::5, ::5]

    detector = CCD(
        geometry=CCDGeometry(row=20, col=2),
        material=Material(),
        environment=Environment(),
        characteristics=CCDCharacteristics(fwc=100_000),
    )

    detector.pixel.array = pixel_2d

    return detector


def test_remove_one_traps():
    """Test arctic model without numba."""

    instant_traps = [{"density": 90.0, "release_timescale": 1.2}]
    num_iterations = 5

    detector1 = create_detector()  # type: CCD
    detector2 = create_detector()  # type: CCD

    # Use library 'arcticpy'
    arctic_remove(
        detector=detector1,
        well_fill_power=0.8,
        instant_traps=instant_traps,
        num_iterations=num_iterations,
    )

    # Use optimized 'arcticpy'
    arctic_remove_numba(
        detector=detector2,
        well_fill_power=0.8,
        instant_traps=instant_traps,
        num_iterations=num_iterations,
    )

    np.testing.assert_almost_equal(detector1.pixel.array, detector2.pixel.array)


def test_remove_five_traps():
    """Test arctic model without numba."""

    instant_traps = [
        {"density": 91.98081597, "release_timescale": 2.07392977},
        {"density": 100.32049957, "release_timescale": 0.75635299},
        {"density": 103.70445648, "release_timescale": 1.48364189},
        {"density": 100.76309597, "release_timescale": 0.70015936},
        {"density": 104.31871946, "release_timescale": 1.30312337},
    ]

    detector1 = create_detector()  # type: CCD
    detector2 = create_detector()  # type: CCD

    # Use library 'arcticpy'
    arctic_remove(
        detector=detector1,
        well_fill_power=0.8,
        instant_traps=instant_traps,
        num_iterations=5,
    )

    # Use optimized 'arcticpy'
    arctic_remove_numba(
        detector=detector2,
        well_fill_power=0.8,
        instant_traps=instant_traps,
        num_iterations=5,
    )

    np.testing.assert_almost_equal(detector1.pixel.array, detector2.pixel.array)

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import arcticpy as ac
import numpy as np
import pytest

from pyxel.models.charge_transfer.arctic_without_numba import CCD as CCD_no_numba
from pyxel.models.charge_transfer.arctic_without_numba import ROE as ROE_no_numba
from pyxel.models.charge_transfer.arctic_without_numba import (
    TrapsInstantCapture as TrapsInstantCapture_no_numba,
)
from pyxel.models.charge_transfer.arctic_without_numba import (
    remove_cti as remove_cti_no_numba,
)


@pytest.fixture
def pixel_2d() -> np.ndarray:
    """Create a valid 2D image."""
    data_2d = np.zeros((100, 10), dtype=int)

    data_2d[25:50, :] = 100
    data_2d[75:, :] = 100

    return data_2d[::5, :5]


@pytest.fixture
def valid_image_removed_one_trap(pixel_2d: np.ndarray) -> np.ndarray:
    """Generate an image with removed CTI with 1 trap."""
    # Input parameters
    well_fill_power = 0.8
    fwc = 100_000

    trap_1_density = 100.0
    trap_1_release_timescale = 1.2

    image_2d = np.asarray(pixel_2d, dtype=float)

    ccd = ac.CCD(well_fill_power=well_fill_power, full_well_depth=fwc)
    roe = ac.ROE()

    traps = [
        ac.Trap(density=trap_1_density, release_timescale=trap_1_release_timescale)
    ]

    image_cti_removed = ac.remove_cti(
        image=image_2d,
        iterations=5,
        parallel_traps=traps,
        parallel_ccd=ccd,
        parallel_roe=roe,
        parallel_express=0,
    )

    return image_cti_removed


def test_remove_cti_no_numba(
    pixel_2d: np.ndarray, valid_image_removed_one_trap: np.ndarray
):
    """Test arctic model without numba."""
    # Input parameters
    well_fill_power = 0.8
    fwc = 100_000

    trap_1_density = 100.0
    trap_1_release_timescale = 1.2

    image_2d = np.asarray(pixel_2d, dtype=float)

    ccd = CCD_no_numba(
        n_phases=1,
        fraction_of_traps_per_phase=np.array([1.0], dtype=np.float64),
        full_well_depth=np.array([fwc], dtype=np.float64),
        well_notch_depth=np.array([0.0], dtype=np.float64),
        well_fill_power=np.array([well_fill_power], dtype=np.float64),
        well_bloom_level=np.array([fwc]),
    )

    parallel_roe = ROE_no_numba(dwell_times=np.array([1.0], dtype=np.float64))

    n_traps = 1
    traps = TrapsInstantCapture_no_numba(
        density_1d=np.array([trap_1_density]),
        release_timescale_1d=np.array([trap_1_release_timescale]),
        surface_1d=np.array([False] * n_traps, dtype=np.bool_),
    )

    traps_lst = []
    traps_lst.append(traps)

    image_cti_removed = remove_cti_no_numba(
        image_2d=image_2d,
        iterations=5,
        parallel_traps=traps_lst,
        parallel_ccd=ccd,
        parallel_roe=parallel_roe,
        parallel_express=0,
        # serial_roe=serial_roe,
    )

    np.testing.assert_equal(image_cti_removed, valid_image_removed_one_trap)

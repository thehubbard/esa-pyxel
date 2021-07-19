#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import arcticpy as ac
import numpy as np
import pytest
from numba.typed import List

from pyxel.models.charge_transfer.arctic_with_numba_02 import CCD as CCD_numba
from pyxel.models.charge_transfer.arctic_with_numba_02 import ROE as ROE_numba
from pyxel.models.charge_transfer.arctic_with_numba_02 import (
    TrapsInstantCapture as TrapsInstantCapture_numba,
)
from pyxel.models.charge_transfer.arctic_with_numba_02 import add_cti as add_cti_numba
from pyxel.models.charge_transfer.arctic_with_numba_02 import (
    remove_cti as remove_cti_numba,
)
from pyxel.models.charge_transfer.arctic_without_numba import CCD as CCD_no_numba
from pyxel.models.charge_transfer.arctic_without_numba import ROE as ROE_no_numba
from pyxel.models.charge_transfer.arctic_without_numba import (
    TrapsInstantCapture as TrapsInstantCapture_no_numba,
)
from pyxel.models.charge_transfer.arctic_without_numba import (
    add_cti as add_cti_no_numba,
)
from pyxel.models.charge_transfer.arctic_without_numba import (
    remove_cti as remove_cti_no_numba,
)


@pytest.fixture
def pixel_2d() -> np.ndarray:
    """Create a valid 2D image."""
    return np.array(
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [100, 100],
            [100, 100],
            [100, 100],
            [100, 100],
            [100, 100],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [100, 100],
            [100, 100],
            [100, 100],
            [100, 100],
            [100, 100],
        ],
        dtype=int,
    )


@pytest.fixture
def valid_image_added_five_traps(pixel_2d: np.ndarray) -> np.ndarray:
    """Generate an image with added CTI with 1 trap."""
    # Input parameters
    well_fill_power = 0.8
    fwc = 100_000

    trap_1_density = 91.98
    trap_2_density = 100.32
    trap_3_density = 103.70
    trap_4_density = 100.76
    trap_5_density = 104.31

    trap_1_release_timescale = 2.07
    trap_2_release_timescale = 0.75
    trap_3_release_timescale = 1.48
    trap_4_release_timescale = 0.70
    trap_5_release_timescale = 1.30

    image_2d = np.asarray(pixel_2d, dtype=float)

    ccd = ac.CCD(well_fill_power=well_fill_power, full_well_depth=fwc)
    roe = ac.ROE()

    traps = [
        ac.Trap(density=trap_1_density, release_timescale=trap_1_release_timescale),
        ac.Trap(density=trap_2_density, release_timescale=trap_2_release_timescale),
        ac.Trap(density=trap_3_density, release_timescale=trap_3_release_timescale),
        ac.Trap(density=trap_4_density, release_timescale=trap_4_release_timescale),
        ac.Trap(density=trap_5_density, release_timescale=trap_5_release_timescale),
    ]

    image_cti_added = ac.add_cti(
        image=image_2d,
        parallel_traps=traps,
        parallel_ccd=ccd,
        parallel_roe=roe,
        parallel_express=0,
    )

    return image_cti_added


def test_add_cti_no_numba_five_traps(
    pixel_2d: np.ndarray, valid_image_added_five_traps: np.ndarray
):
    """Test arctic model without numba."""
    # Input parameters
    well_fill_power = 0.8
    fwc = 100_000

    trap_1_density = 91.98
    trap_2_density = 100.32
    trap_3_density = 103.70
    trap_4_density = 100.76
    trap_5_density = 104.31

    trap_1_release_timescale = 2.07
    trap_2_release_timescale = 0.75
    trap_3_release_timescale = 1.48
    trap_4_release_timescale = 0.70
    trap_5_release_timescale = 1.30

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

    n_traps = 5
    traps = TrapsInstantCapture_no_numba(
        density_1d=np.array(
            [
                trap_1_density,
                trap_2_density,
                trap_3_density,
                trap_4_density,
                trap_5_density,
            ]
        ),
        release_timescale_1d=np.array(
            [
                trap_1_release_timescale,
                trap_2_release_timescale,
                trap_3_release_timescale,
                trap_4_release_timescale,
                trap_5_release_timescale,
            ]
        ),
        surface_1d=np.array([False] * n_traps, dtype=np.bool_),
    )

    traps_lst = []
    traps_lst.append(traps)

    image_cti_added = add_cti_no_numba(
        image_2d=image_2d,
        parallel_traps=traps_lst,
        parallel_ccd=ccd,
        parallel_roe=parallel_roe,
        parallel_express=0,
        # serial_roe=serial_roe,
    )

    np.testing.assert_equal(image_cti_added, valid_image_added_five_traps)

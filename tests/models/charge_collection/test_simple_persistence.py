#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import typing as t

import numpy as np
import pytest

from pyxel.detectors import CMOS, CMOSCharacteristics, CMOSGeometry, Environment
from pyxel.models.charge_collection import simple_persistence


@pytest.fixture
def cmos_5x10() -> CMOS:
    """Create a valid CCD detector."""
    detector = CMOS(
        geometry=CMOSGeometry(
            row=5,
            col=10,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(),
        characteristics=CMOSCharacteristics(),
    )

    return detector


def test_simple_persistence(cmos_5x10: CMOS):
    """Test model 'simple_persistence'."""
    detector = cmos_5x10

    # No persistence
    assert "persistence" not in detector._memory

    simple_persistence(
        detector=detector,
        trap_timeconstants=[1.0, 3.0],
        trap_densities=[2.0, 4.0],
    )

    assert "persistence" in detector._memory
    assert len(detector._memory["persistence"]) == 2

    # With persistence
    simple_persistence(
        detector=detector,
        trap_timeconstants=[1.0],
        trap_densities=[2.0],
    )

    assert "persistence" in detector._memory
    assert len(detector._memory["persistence"]) == 2


@pytest.mark.parametrize(
    "trap_timeconstants, trap_densities, exp_error, exp_msg",
    [
        pytest.param(
            [],
            [],
            ValueError,
            "Expecting at least one 'trap_timeconstants' and 'trap_densities'",
            id="no elements",
        ),
        pytest.param(
            [1.0],
            [2.0, 3.0],
            ValueError,
            "Expecting same number of elements for parameters",
            id="not same number of elements",
        ),
    ],
)
def test_simple_persistence_bad_inputs(
    cmos_5x10: CMOS,
    trap_timeconstants,
    trap_densities,
    exp_error,
    exp_msg,
):
    """Test model 'simple_persistence' with bad inputs."""
    detector = cmos_5x10

    with pytest.raises(exp_error, match=exp_msg):
        simple_persistence(
            detector=detector,
            trap_timeconstants=trap_timeconstants,
            trap_densities=trap_densities,
        )

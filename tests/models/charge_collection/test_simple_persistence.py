#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import typing as t

import numpy as np
import pytest

from pyxel.detectors import (
    CMOS,
    CMOSCharacteristics,
    CMOSGeometry,
    Environment,
    Material,
)
from pyxel.models.charge_collection import simple_persistence


@pytest.fixture
def cmos_no_persistence_5x10() -> CMOS:
    """Create a valid CCD detector."""
    detector = CMOS(
        geometry=CMOSGeometry(
            row=5,
            col=10,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        material=Material(),
        environment=Environment(),
        characteristics=CMOSCharacteristics(),
    )

    return detector


@pytest.mark.parametrize(
    "trap_timeconstants, trap_densities",
    [
        ((1.0,), (1.0,)),
    ],
)
def test_simple_persistence(
    cmos_no_persistence_5x10: CMOS,
    trap_timeconstants: t.List[float],
    trap_densities: t.List[float],
):
    """Test model 'simple_persistence'."""
    detector = cmos_no_persistence_5x10
    assert "persistence" not in detector._memory

    simple_persistence(
        detector=detector,
        trap_timeconstants=trap_timeconstants,
        trap_densities=trap_densities,
    )

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
    CCD,
    CMOS,
    CCDCharacteristics,
    CCDGeometry,
    CMOSCharacteristics,
    CMOSGeometry,
    Environment,
    Material,
)
from pyxel.models.charge_measurement import output_node_noise


@pytest.fixture
def ccd_5x10() -> CCD:
    """Create a valid CCD detector."""
    return CCD(
        geometry=CCDGeometry(
            row=5,
            col=10,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        material=Material(),
        environment=Environment(),
        characteristics=CCDCharacteristics(),
    )


@pytest.fixture
def cmos_5x10() -> CMOS:
    """Create a valid CMOS detector."""
    return CMOS(
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


@pytest.mark.parametrize("detector_type", ["ccd", "cmos"])
@pytest.mark.parametrize(
    "std_deviation",
    [pytest.param(1.0, id="std positive"), pytest.param(0.0, id="std null")],
)
def test_output_node_noise(
    ccd_5x10: CCD,
    cmos_5x10: CMOS,
    detector_type: str,
    std_deviation: float,
):
    """Test model 'output_node_noise' with valid inputs."""
    if detector_type == "ccd":
        detector = ccd_5x10
    elif detector_type == "cmos":
        detector = cmos_5x10
    else:
        raise NotImplementedError

    output_node_noise(detector=detector, std_deviation=std_deviation)


@pytest.mark.parametrize("detector_type", ["ccd", "cmos"])
def test_output_node_noise_bad_std(ccd_5x10: CCD, cmos_5x10: CMOS, detector_type: str):
    """Test model 'output_node_noise' with invalid input(s)."""
    if detector_type == "ccd":
        detector = ccd_5x10
    elif detector_type == "cmos":
        detector = cmos_5x10
    else:
        raise NotImplementedError

    with pytest.raises(ValueError, match="'std_deviation' must be positive."):
        output_node_noise(detector=detector, std_deviation=-1.0)

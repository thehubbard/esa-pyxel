#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import typing as t

import pytest

from pyxel.detectors import CCD, CCDCharacteristics, CCDGeometry, Environment, Material
from pyxel.models.readout_electronics import sar_adc


@pytest.fixture
def ccd_10x3() -> CCD:
    """Create a valid CCD detector."""
    return CCD(
        geometry=CCDGeometry(
            row=10,
            col=3,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        material=Material(),
        environment=Environment(),
        characteristics=CCDCharacteristics(),
    )


@pytest.mark.parametrize(
    "adc_bits, range_volt",
    [
        pytest.param(16, (0.0, 5.0)),
        pytest.param(32, (0.0, 5.0)),
        pytest.param(8, (5.0, 10.0)),
    ],
)
def test_sar_adc(ccd_10x3: CCD, adc_bits: int, range_volt: t.Tuple[int, int]):
    """Test model 'sar_adc' with valid inputs."""
    sar_adc(detector=ccd_10x3, adc_bits=adc_bits, range_volt=range_volt)


@pytest.mark.parametrize(
    "adc_bits, range_volt, exp_exc, exp_error",
    [
        pytest.param(
            0,
            (0.0, 5.0),
            ValueError,
            "Expecting a strictly positive value for 'adc_bits'",
        ),
        pytest.param(
            -1,
            (0.0, 5.0),
            ValueError,
            "Expecting a strictly positive value for 'adc_bits'",
        ),
    ],
)
def test_sar_adc_bad_inputs(ccd_10x3: CCD, adc_bits, range_volt, exp_exc, exp_error):
    """Test model 'sar_adc' with bad inputs."""
    with pytest.raises(exp_exc, match=exp_error):
        sar_adc(detector=ccd_10x3, adc_bits=adc_bits, range_volt=range_volt)

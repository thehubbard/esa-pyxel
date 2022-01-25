#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
import pytest
import typing as t
from pyxel.detectors import CCD, CCDCharacteristics, CCDGeometry, Environment
from pyxel.models.readout_electronics import simple_adc


@pytest.fixture
def ccd_3x3() -> CCD:
    """Create a valid CCD detector."""
    detector = CCD(
        geometry=CCDGeometry(
            row=3,
            col=3,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(),
        characteristics=CCDCharacteristics(),
    )
    detector.characteristics.adc_bit_resolution=16
    detector.characteristics.adc_voltage_range=(0., 5.)
    return detector


@pytest.mark.parametrize("data_type", ["uint16", "uint32", "uint64", "uint"])
def test_simple_adc(ccd_3x3: CCD, data_type: str):
    """Test model 'simple_adc'."""
    simple_adc(detector=ccd_3x3, data_type=data_type)


@pytest.mark.parametrize(
    "voltage_range, bit_resolution, data_type, exp_exc, exp_error",
    [
        ([0., 5.], 16, "float", TypeError, "Expecting a signed/unsigned integer"),
        ([0., 5.], 16, "float32", TypeError, "Expecting a signed/unsigned integer"),
        ([0., 5.], 16, "float64", TypeError, "Expecting a signed/unsigned integer"),
        ([0., 5.], 16, "uint8", TypeError, "Expected type of Image array is uint64"),
        ([0., 5.], 16, "int32", TypeError, "Expected type of Image array is uint64"),
        ([0., 5.], 16, "int64", TypeError, "Expected type of Image array is uint64"),
        ([0., 5.], 16, "int", TypeError, "Expected type of Image array is uint64"),
        ([0., 5., 6.], 16, "int", ValueError, "Voltage range must have length of 2."),
        ([0., 5.], 100, "int", ValueError, "'adc_bit_resolution' must be between 4 and 64."),
    ],
)
def test_simple_adc_wrong_data_type(
    ccd_3x3: CCD, voltage_range: t.Tuple[float, float], bit_resolution: int, data_type: str, exp_exc, exp_error
):
    """Test model 'simple_adc' with wrong 'data_type'."""
    with pytest.raises(exp_exc, match=exp_error):
        simple_adc(detector=ccd_3x3, voltage_range=voltage_range, bit_resolution=bit_resolution, data_type=data_type)
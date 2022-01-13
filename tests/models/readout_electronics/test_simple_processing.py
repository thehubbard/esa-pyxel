#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
import numpy as np
import pytest

from pyxel.detectors import CCD, CCDCharacteristics, CCDGeometry, Environment, Material
from pyxel.models.readout_electronics import simple_processing


@pytest.fixture
def ccd_3x3() -> CCD:
    """Create a valid CCD detector."""
    return CCD(
        geometry=CCDGeometry(
            row=3,
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
    "gain_adc",
    [None, 2.0],
)
def test_simple_processing_valid(ccd_3x3: CCD, gain_adc: float):
    """Test model 'simple_processing' with valid inputs."""
    detector = ccd_3x3
    detector.characteristics.adc_gain = 2.0
    detector.signal.array = np.ones((3, 3), dtype=float)
    start = detector.signal.array
    end = start * 2.0

    simple_processing(detector=detector, gain_adc=gain_adc)

    np.testing.assert_array_almost_equal(end, detector.image.array)

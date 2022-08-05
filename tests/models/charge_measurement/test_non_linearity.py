#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
import typing as t

import numpy as np
import pytest

from pyxel.detectors import CCD, CCDCharacteristics, CCDGeometry, Environment
from pyxel.models.charge_measurement import output_node_linearity_poly


@pytest.fixture
def ccd_5x5() -> CCD:
    """Create a valid CCD detector."""
    return CCD(
        geometry=CCDGeometry(
            row=5,
            col=5,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(),
        characteristics=CCDCharacteristics(),
    )


@pytest.mark.parametrize(
    "coefficients",
    [
        pytest.param([0, 1, 0.9]),
        pytest.param([5, 0.5, 0.9, 0.8]),
        pytest.param([3]),
        pytest.param([0, 3]),
    ],
)
def test_non_linearity_valid(ccd_5x5: CCD, coefficients: t.Sequence):
    """Test model 'non_linearity' with valid inputs."""
    detector = ccd_5x5
    detector.signal.array = np.ones(detector.signal.shape)
    output_node_linearity_poly(detector=detector, coefficients=coefficients)


@pytest.mark.parametrize(
    "coefficients, exp_exc, exp_error",
    [
        pytest.param(
            [], ValueError, "Length of coefficient list should be more than 0."
        ),
        pytest.param(
            [0, -10],
            ValueError,
            "Signal array contains negative values after applying non-linearity model!",
        ),
    ],
)
def test_non_linearity_invalid(
    ccd_5x5: CCD, coefficients: t.Sequence, exp_exc, exp_error
):
    """Test model 'non_linearity' with valid inputs."""
    detector = ccd_5x5
    detector.signal.array = np.ones(detector.signal.shape)
    with pytest.raises(exp_exc, match=exp_error):
        output_node_linearity_poly(detector=detector, coefficients=coefficients)

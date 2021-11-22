#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import pytest

from pyxel.detectors import (
    CMOS,
    CMOSCharacteristics,
    CMOSGeometry,
    Environment,
    Material,
)
from pyxel.models.readout_electronics import non_destructive_readout


@pytest.fixture
def cmos_10x10_with_readout() -> CMOS:
    """Create a valid CMOS detector."""
    detector = CMOS(
        geometry=CMOSGeometry(
            row=3,
            col=3,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        material=Material(),
        environment=Environment(),
        characteristics=CMOSCharacteristics(),
    )

    detector.set_readout(num_steps=10, start_time=0.0, end_time=100, ndreadout=True)

    return detector


@pytest.mark.parametrize(
    "mode, fowler_samples",
    [
        ("uncorrelated", None),
        ("CDS", None),
        ("Fowler-N", 1),
        ("UTR", None),
    ],
)
def test_non_destructive_readout(cmos_10x10_with_readout, mode, fowler_samples):
    """Test function 'non_destructive_readout'."""
    non_destructive_readout(
        detector=cmos_10x10_with_readout, mode=mode, fowler_samples=fowler_samples
    )


@pytest.mark.parametrize(
    "mode, fowler_samples, exp_exc, exp_msg",
    [
        pytest.param(
            "uncorrelated",
            1,
            ValueError,
            "Parameter 'fowler_samples' can only be used for mode 'Fowler-N'",
            id="Uncorrelated + fowler_sample",
        ),
        pytest.param(
            "CDS",
            1,
            ValueError,
            "Parameter 'fowler_samples' can only be used for mode 'Fowler-N'",
            id="CDS + fowler_sample",
        ),
        pytest.param(
            "UTR",
            1,
            ValueError,
            "Parameter 'fowler_samples' can only be used for mode 'Fowler-N'",
            id="UTR + fowler_sample",
        ),
        pytest.param(
            "Fowler-N",
            None,
            ValueError,
            "Missing parameter 'fowler_samples' for mode 'Fowler-N'",
            id="Missing fowler_sample",
        ),
        pytest.param(
            "Fowler-N",
            -1,
            ValueError,
            "Parameter 'fowler_samples' must be > 1",
            id="Negative fowler_sample",
        ),
        pytest.param("cds", None, ValueError, "Unknown mode", id="Bad mode 1"),
        pytest.param("dummy", None, ValueError, "Unknown mode", id="Bad mode 2"),
    ],
)
def test_non_destructive_readout_bad_inputs(
    cmos_10x10_with_readout, mode, fowler_samples, exp_exc, exp_msg
):
    """Test function 'non_destructive_readout' with bad inputs.."""
    with pytest.raises(exp_exc, match=exp_msg):
        non_destructive_readout(
            detector=cmos_10x10_with_readout, mode=mode, fowler_samples=fowler_samples
        )


@pytest.mark.parametrize(
    "ndreadout, times_linear, exp_exc, exp_msg",
    [
        pytest.param(
            False,
            True,
            ValueError,
            "Detector is must have a non-destructive readout and must be dynamic",
            id="Missing non-destructive readout",
        ),
        pytest.param(
            True, False, ValueError, "Detector's time must be linear", id="Not linear"
        ),
    ],
)
def test_cmos_without_destructive_mode(
    ndreadout: bool, times_linear: bool, exp_exc, exp_msg
):
    """Test function 'non_destructive_mode' without a detector properly initialized."""
    detector = CMOS(
        geometry=CMOSGeometry(
            row=3,
            col=3,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        material=Material(),
        environment=Environment(),
        characteristics=CMOSCharacteristics(),
    )
    detector.set_readout(
        num_steps=10,
        start_time=0.0,
        end_time=100,
        ndreadout=ndreadout,
        times_linear=times_linear,
    )

    with pytest.raises(exp_exc, match=exp_msg):
        non_destructive_readout(detector=detector, mode="uncorrelated")

#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.


from typing import Union

import numpy as np
import pytest

from pyxel.detectors import (
    APD,
    CCD,
    CMOS,
    APDCharacteristics,
    APDGeometry,
    CCDGeometry,
    Characteristics,
    CMOSGeometry,
    Environment,
)
from pyxel.models.charge_measurement import (
    output_node_noise,
    output_node_noise_cmos,
    readout_noise_saphira,
)


@pytest.fixture
def ccd_2x3() -> CCD:
    """Create a valid CCD detector."""
    detector = CCD(
        geometry=CCDGeometry(
            row=2,
            col=3,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(),
        characteristics=Characteristics(),
    )
    detector.signal.array = np.zeros(detector.geometry.shape, dtype=float)
    return detector


@pytest.fixture
def cmos_2x3() -> CMOS:
    """Create a valid CMOS detector."""
    detector = CMOS(
        geometry=CMOSGeometry(
            row=2,
            col=3,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(),
        characteristics=Characteristics(charge_to_volt_conversion=0.01),
    )
    detector.signal.array = np.zeros(detector.geometry.shape, dtype=float)
    return detector


@pytest.fixture
def apd_2x3() -> APD:
    """Create a valid CCD detector."""
    detector = APD(
        geometry=APDGeometry(
            row=2,
            col=3,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(),
        characteristics=APDCharacteristics(
            quantum_efficiency=1.0,
            adc_voltage_range=(0.0, 10.0),
            adc_bit_resolution=16,
            full_well_capacity=100000,
            avalanche_gain=1.0,
            pixel_reset_voltage=5.0,
            roic_gain=0.8,
        ),
    )
    detector.signal.array = np.zeros(detector.geometry.shape, dtype=float)
    return detector


@pytest.mark.parametrize("detector_type", ["ccd", "cmos"])
@pytest.mark.parametrize(
    "std_deviation",
    [pytest.param(1.0, id="std positive"), pytest.param(0.0, id="std null")],
)
def test_output_node_noise(
    ccd_2x3: CCD,
    cmos_2x3: CMOS,
    detector_type: str,
    std_deviation: float,
):
    """Test model 'output_node_noise' with valid inputs."""
    if detector_type == "ccd":
        detector: Union[CCD, CMOS] = ccd_2x3
    elif detector_type == "cmos":
        detector = cmos_2x3
    else:
        raise NotImplementedError

    seed = 12345
    rng = np.random.default_rng(seed=seed)
    signal_2d = rng.random(size=(2, 3), dtype=float)
    detector.signal.array = signal_2d.copy()

    output_node_noise(
        detector=detector,
        std_deviation=std_deviation,
        seed=seed,
    )

    new_signal = detector.signal.array

    if std_deviation == 1.0:
        exp_signal = np.array(
            [[0.02262836, 0.79570168, 0.27792674], [0.12052437, 2.35689012, 1.72621976]]
        )
    else:
        exp_signal = signal_2d

    np.testing.assert_allclose(actual=new_signal, desired=exp_signal, rtol=1e-6)


@pytest.mark.parametrize("detector_type", ["ccd", "cmos"])
def test_output_node_noise_bad_std(ccd_2x3: CCD, cmos_2x3: CMOS, detector_type: str):
    """Test model 'output_node_noise' with invalid input(s)."""
    if detector_type == "ccd":
        detector: Union[CCD, CMOS] = ccd_2x3
    elif detector_type == "cmos":
        detector = cmos_2x3
    else:
        raise NotImplementedError

    with pytest.raises(ValueError, match="'std_deviation' must be positive."):
        output_node_noise(detector=detector, std_deviation=-1.0)


def test_output_node_noise_cmos(cmos_2x3: CMOS):
    """Test model 'output_node_noise_cmos' with valid inputs."""
    seed = 12345
    rng = np.random.default_rng(seed=seed)

    detector = cmos_2x3
    detector.signal.array = rng.random(size=(2, 3), dtype=float)

    output_node_noise_cmos(
        detector=detector,
        readout_noise=1.0,
        readout_noise_std=0.1,
        seed=seed,
    )

    new_signal = detector.signal.array

    exp_signal = np.array(
        [[0.228246, 0.319711, 0.804656], [0.688026, 0.403161, 0.318046]]
    )
    np.testing.assert_allclose(actual=new_signal, desired=exp_signal, rtol=1e-5)


def test_output_node_noise_with_ccd(ccd_2x3: CCD):
    """Test model 'output_node_noise_ccd' with a 'CCD'."""
    detector = ccd_2x3

    with pytest.raises(TypeError, match="Expecting a 'CMOS' detector object."):
        output_node_noise_cmos(
            detector=detector,
            readout_noise=1.0,
            readout_noise_std=2.0,
        )


def test_output_node_noise_invalid_noise(cmos_2x3: CMOS):
    """Test model 'output_node_noise_ccd' with a 'CCD'."""
    detector = cmos_2x3

    with pytest.raises(ValueError, match="'readout_noise_std' must be positive."):
        output_node_noise_cmos(
            detector=detector,
            readout_noise=1.0,
            readout_noise_std=-1.0,
        )


def test_readout_noise_saphira_with_ccd(ccd_2x3: CCD):
    """Test model 'readout_noise_saphira' with a 'CCD'."""
    detector = ccd_2x3

    with pytest.raises(TypeError, match="Expecting a 'APD' detector object."):
        readout_noise_saphira(
            detector=detector,
            roic_readout_noise=0.1,
            controller_noise=0.1,
        )


def test_readout_noise_saphira(apd_2x3: APD):
    """Test model 'readout_noise_saphira' with valid inputs."""
    detector = apd_2x3

    readout_noise_saphira(
        detector=detector,
        roic_readout_noise=0.1,
        controller_noise=0.1,
    )

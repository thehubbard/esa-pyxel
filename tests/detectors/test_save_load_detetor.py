#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.


import typing as t
from pathlib import Path

import pytest

from pyxel.detectors import (
    APD,
    CCD,
    CMOS,
    MKID,
    APDCharacteristics,
    APDGeometry,
    CCDCharacteristics,
    CCDGeometry,
    CMOSCharacteristics,
    CMOSGeometry,
    Detector,
    Environment,
    MKIDCharacteristics,
    MKIDGeometry,
)


@pytest.fixture(
    params=(
        "ccd_basic",
        "ccd_100x120",
        "cmos_basic",
        "cmos_100x120",
        "mkid_basic",
        "mkid_100x120",
        "apd_basic",
        "apd_100x120",
    )
)
def detector(request) -> t.Union[CCD, CMOS, MKID, APD]:

    if request.param == "ccd_basic":
        return CCD(
            geometry=CCDGeometry(row=4, col=5),
            environment=Environment(),
            characteristics=CCDCharacteristics(),
        )
    elif request.param == "ccd_100x120":
        return CCD(
            geometry=CCDGeometry(
                row=100,
                col=120,
                total_thickness=123.1,
                pixel_horz_size=12.4,
                pixel_vert_size=34.5,
            ),
            environment=Environment(temperature=100.1),
            characteristics=CCDCharacteristics(
                quantum_efficiency=0.1,
                charge_to_volt_conversion=0.2,
                pre_amplification=3.3,
                full_well_capacity=10,
            ),
        )
    elif request.param == "cmos_basic":
        return CMOS(
            geometry=CMOSGeometry(row=4, col=5),
            environment=Environment(),
            characteristics=CMOSCharacteristics(),
        )
    elif request.param == "cmos_100x120":
        return CMOS(
            geometry=CMOSGeometry(
                row=100,
                col=120,
                total_thickness=123.1,
                pixel_horz_size=12.4,
                pixel_vert_size=34.5,
            ),
            environment=Environment(temperature=100.1),
            characteristics=CMOSCharacteristics(
                quantum_efficiency=0.1,
                charge_to_volt_conversion=0.2,
                pre_amplification=3.3,
                full_well_capacity=10,
            ),
        )
    elif request.param == "mkid_basic":
        return MKID(
            geometry=MKIDGeometry(row=4, col=5),
            environment=Environment(),
            characteristics=MKIDCharacteristics(),
        )
    elif request.param == "mkid_100x120":
        return MKID(
            geometry=MKIDGeometry(
                row=100,
                col=120,
                total_thickness=123.1,
                pixel_horz_size=12.4,
                pixel_vert_size=34.5,
            ),
            environment=Environment(temperature=100.1),
            characteristics=MKIDCharacteristics(
                quantum_efficiency=0.1,
                charge_to_volt_conversion=0.2,
                pre_amplification=3.3,
                full_well_capacity=10,
            ),
        )
    elif request.param == "apd_basic":
        return APD(
            geometry=APDGeometry(row=4, col=5),
            environment=Environment(),
            characteristics=APDCharacteristics(),
        )
    elif request.param == "apd_100x120":
        return APD(
            geometry=APDGeometry(
                row=100,
                col=120,
                total_thickness=123.1,
                pixel_horz_size=12.4,
                pixel_vert_size=34.5,
            ),
            environment=Environment(temperature=100.1),
            characteristics=APDCharacteristics(
                quantum_efficiency=0.1,
                full_well_capacity=10,
                adc_bit_resolution=16,
                adc_voltage_range=(0.0, 5.0),
                avalanche_gain=1.0,
                pixel_reset_voltage=12.0,
                common_voltage=3.0,
                roic_gain=4.1,
            ),
        )
    else:
        raise NotImplementedError


def test_to_from_hdf5(detector: t.Union[CCD, CMOS, MKID, APD], tmp_path: Path):
    """Test methods `Detector.to_hdf5' and `Detector.from_hdf5`."""
    filename = tmp_path / "ccd.h5"

    # Save to 'hdf5'
    detector.to_hdf5(filename)

    # Load from 'hdf5
    new_detector = Detector.from_hdf5(filename)

    # Comparse
    assert new_detector == detector

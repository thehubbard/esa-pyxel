#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from pyxel.detectors import (
    CMOS,
    CMOSCharacteristics,
    CMOSGeometry,
    Detector,
    Environment,
)


@pytest.fixture
def valid_cmos() -> CMOS:
    """Create a valid `CMOS`."""
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
            full_well_capacity=4.4,
        ),
    )


@pytest.mark.parametrize(
    "other_obj, is_equal",
    [
        pytest.param(None, False, id="None"),
        pytest.param(
            CMOS(
                geometry=CMOSGeometry(row=100, col=120),
                environment=Environment(),
                characteristics=CMOSCharacteristics(),
            ),
            False,
            id="Empty 'CMOS'",
        ),
        pytest.param(
            CMOS(
                geometry=CMOSGeometry(
                    row=100,
                    col=120,
                    total_thickness=123.1,
                    pixel_horz_size=12.4,
                    pixel_vert_size=34.5,
                ),
                environment=Environment(),
                characteristics=CMOSCharacteristics(),
            ),
            False,
            id="Almost same parameters, different class",
        ),
        pytest.param(
            CMOS(
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
                    full_well_capacity=4.4,
                ),
            ),
            True,
            id="Same parameters, same class",
        ),
    ],
)
def test_is_equal(valid_cmos: CMOS, other_obj, is_equal):
    """Test equality statement for `CMOS`."""

    if is_equal:
        assert valid_cmos == other_obj
    else:
        assert valid_cmos != other_obj


def comparison(dct, other_dct):
    assert set(dct) == set(other_dct) == {"version", "type", "data", "properties"}
    assert dct["version"] == other_dct["version"]
    assert dct["type"] == other_dct["type"]
    assert dct["properties"] == other_dct["properties"]

    assert (
        set(dct["data"])
        == set(other_dct["data"])
        == {"photon", "pixel", "signal", "image", "charge"}
    )
    np.testing.assert_equal(dct["data"]["photon"], other_dct["data"]["photon"])
    np.testing.assert_equal(dct["data"]["pixel"], other_dct["data"]["pixel"])
    np.testing.assert_equal(dct["data"]["signal"], other_dct["data"]["signal"])
    np.testing.assert_equal(dct["data"]["image"], other_dct["data"]["image"])

    assert (
        set(dct["data"]["charge"])
        == set(other_dct["data"]["charge"])
        == {"array", "frame"}
    )
    np.testing.assert_equal(
        dct["data"]["charge"]["array"], other_dct["data"]["charge"]["array"]
    )
    pd.testing.assert_frame_equal(
        dct["data"]["charge"]["frame"], other_dct["data"]["charge"]["frame"]
    )


@pytest.mark.parametrize("klass", [CMOS, Detector])
@pytest.mark.parametrize(
    "obj, exp_dict",
    [
        (
            CMOS(
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
                    full_well_capacity=4.4,
                ),
            ),
            {
                "version": 1,
                "type": "cmos",
                "properties": {
                    "geometry": {
                        "row": 100,
                        "col": 120,
                        "total_thickness": 123.1,
                        "pixel_horz_size": 12.4,
                        "pixel_vert_size": 34.5,
                    },
                    "environment": {"temperature": 100.1},
                    "characteristics": {
                        "quantum_efficiency": 0.1,
                        "charge_to_volt_conversion": 0.2,
                        "pre_amplification": 3.3,
                        "full_well_capacity": 4.4,
                    },
                },
                "data": {
                    "photon": np.zeros(shape=(100, 120)),
                    "pixel": np.zeros(shape=(100, 120)),
                    "signal": np.zeros(shape=(100, 120)),
                    "image": np.zeros(shape=(100, 120)),
                    "charge": {
                        "array": np.zeros(shape=(100, 120)),
                        "frame": pd.DataFrame(
                            columns=[
                                "charge",
                                "number",
                                "init_energy",
                                "energy",
                                "init_pos_ver",
                                "init_pos_hor",
                                "init_pos_z",
                                "position_ver",
                                "position_hor",
                                "position_z",
                                "velocity_ver",
                                "velocity_hor",
                                "velocity_z",
                            ],
                            dtype=float,
                        ),
                    },
                },
            },
        )
    ],
)
def test_to_and_from_dict(klass, obj, exp_dict):
    """Test methods 'to_dict', 'from_dict'."""
    assert type(obj) == CMOS

    # Convert from `CMOS` to a `dict`
    dct = obj.to_dict()
    comparison(dct, exp_dict)

    # Copy 'exp_dict'
    copied_dct = deepcopy(exp_dict)  # create a new dict
    assert copied_dct is not exp_dict
    comparison(copied_dct, exp_dict)

    # Convert from `dict` to `CMOS`
    other_obj = klass.from_dict(copied_dct)
    assert type(other_obj) == CMOS
    assert obj == other_obj
    assert obj is not other_obj
    comparison(copied_dct, exp_dict)

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
    MKID,
    Detector,
    Environment,
    MKIDCharacteristics,
    MKIDGeometry,
)


@pytest.fixture
def valid_mkid() -> MKID:
    """Create a valid `MKID`."""
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
        ),
    )


@pytest.mark.parametrize(
    "other_obj, is_equal",
    [
        pytest.param(None, False, id="None"),
        pytest.param(
            MKID(
                geometry=MKIDGeometry(row=100, col=120),
                environment=Environment(),
                characteristics=MKIDCharacteristics(),
            ),
            False,
            id="Empty 'MKID'",
        ),
        pytest.param(
            MKID(
                geometry=MKIDGeometry(
                    row=100,
                    col=120,
                    total_thickness=123.1,
                    pixel_horz_size=12.4,
                    pixel_vert_size=34.5,
                ),
                environment=Environment(),
                characteristics=MKIDCharacteristics(),
            ),
            False,
            id="Almost same parameters, different class",
        ),
        pytest.param(
            MKID(
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
                ),
            ),
            True,
            id="Same parameters, same class",
        ),
    ],
)
def test_is_equal(valid_mkid: MKID, other_obj, is_equal):
    """Test equality statement for `MKID`."""

    if is_equal:
        assert valid_mkid == other_obj
    else:
        assert valid_mkid != other_obj


def comparison(dct, other_dct):
    assert set(dct) == set(other_dct) == {"version", "type", "data", "properties"}
    assert dct["version"] == other_dct["version"]
    assert dct["type"] == other_dct["type"]
    assert dct["properties"] == other_dct["properties"]

    assert (
        set(dct["data"])
        == set(other_dct["data"])
        == {"photon", "pixel", "signal", "image", "phase", "charge"}
    )
    np.testing.assert_equal(dct["data"]["photon"], other_dct["data"]["photon"])
    np.testing.assert_equal(dct["data"]["pixel"], other_dct["data"]["pixel"])
    np.testing.assert_equal(dct["data"]["signal"], other_dct["data"]["signal"])
    np.testing.assert_equal(dct["data"]["image"], other_dct["data"]["image"])
    np.testing.assert_equal(dct["data"]["phase"], other_dct["data"]["phase"])

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


@pytest.mark.parametrize("klass", [MKID, Detector])
@pytest.mark.parametrize(
    "obj, exp_dict",
    [
        (
            MKID(
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
                ),
            ),
            {
                "version": 1,
                "type": "mkid",
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
                        "full_well_capacity": None,
                    },
                },
                "data": {
                    "photon": np.zeros(shape=(100, 120)),
                    "pixel": np.zeros(shape=(100, 120)),
                    "signal": np.zeros(shape=(100, 120)),
                    "image": np.zeros(shape=(100, 120)),
                    "phase": np.zeros(shape=(100, 120)),
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
    assert type(obj) == MKID

    # Convert from `MKID` to a `dict`
    dct = obj.to_dict()
    comparison(dct, exp_dict)

    # Copy 'exp_dict'
    copied_dct = deepcopy(exp_dict)  # create a new dict
    assert copied_dct is not exp_dict
    comparison(copied_dct, exp_dict)

    # Convert from `dict` to `MKID`
    other_obj = klass.from_dict(copied_dct)
    assert type(other_obj) == MKID
    assert obj == other_obj
    assert obj is not other_obj
    comparison(copied_dct, exp_dict)

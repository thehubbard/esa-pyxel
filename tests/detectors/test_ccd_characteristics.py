#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from copy import deepcopy

import pytest

from pyxel.detectors import CCDCharacteristics, Characteristics


@pytest.mark.parametrize(
    "qe, eta, sv, amp, a1, a2, fwc, vg, dt, fwc_serial, svg, t, st",
    [
        (0, 0, 0, 0, 0, 0, 0, 0.0, 0, 0, 0.0, 0.0, 0),
        (
            1.0,
            1.0,
            100.0,
            100.0,
            100.0,
            65536,
            10_000_000,
            1.0,
            10.0,
            10_000_000,
            1.0,
            10.0,
            10.0,
        ),
    ],
)
def test_create_valid_ccdcharacteristics(
    qe, eta, sv, amp, a1, a2, fwc, vg, dt, fwc_serial, svg, t, st
):
    """Test when creating a valid `CCDCharacteristics` object."""
    _ = CCDCharacteristics(
        qe=qe,
        eta=eta,
        sv=sv,
        amp=amp,
        a1=a1,
        a2=a2,
        fwc=fwc,
        vg=vg,
        dt=dt,
        fwc_serial=fwc_serial,
        svg=svg,
        t=t,
        st=st,
    )


@pytest.mark.parametrize(
    "other_obj, is_equal",
    [
        pytest.param(None, False, id="None"),
        pytest.param(CCDCharacteristics(), False, id="Empty 'CCDCharacteristics'"),
        pytest.param(CCDCharacteristics(qe=0.1), False, id="Only one parameter"),
        pytest.param(
            Characteristics(
                qe=0.1, eta=0.2, sv=3.3, amp=4.4, a1=5, a2=6, fwc=7, vg=0.8, dt=9.9
            ),
            False,
            id="Wrong type",
        ),
        pytest.param(
            CCDCharacteristics(
                qe=0.1,
                eta=0.2,
                sv=3.3,
                amp=4.4,
                a1=5,
                a2=6,
                fwc=7,
                vg=0.8,
                dt=9.9,
                fwc_serial=10,
                svg=0.11,
                t=0.12,
                st=0.13,
            ),
            True,
            id="Almost same parameters, different class",
        ),
    ],
)
def test_is_equal(other_obj, is_equal):
    """Test equality statement for `CCDCharacteristics`."""
    obj = CCDCharacteristics(
        qe=0.1,
        eta=0.2,
        sv=3.3,
        amp=4.4,
        a1=5,
        a2=6,
        fwc=7,
        vg=0.8,
        dt=9.9,
        fwc_serial=10,
        svg=0.11,
        t=0.12,
        st=0.13,
    )

    if is_equal:
        assert obj == other_obj
    else:
        assert obj != other_obj


@pytest.mark.parametrize(
    "obj, exp_dict",
    [
        (
            CCDCharacteristics(
                qe=0.1,
                eta=0.2,
                sv=3.3,
                amp=4.4,
                a1=5,
                a2=6,
                fwc=7,
                vg=0.8,
                dt=9.9,
                fwc_serial=10,
                svg=0.11,
                t=0.12,
                st=0.13,
            ),
            {
                "qe": 0.1,
                "eta": 0.2,
                "sv": 3.3,
                "amp": 4.4,
                "a1": 5,
                "a2": 6,
                "fwc": 7,
                "vg": 0.8,
                "dt": 9.9,
                "fwc_serial": 10,
                "svg": 0.11,
                "t": 0.12,
                "st": 0.13,
            },
        )
    ],
)
def test_to_and_from_dict(obj, exp_dict):
    """Test methods 'to_dict', 'from_dict'."""
    assert type(obj) == CCDCharacteristics

    # Convert from `CCDCharacteristics` to a `dict`
    dct = obj.to_dict()
    assert dct == exp_dict

    # Copy 'exp_dict'
    copied_dct = deepcopy(exp_dict)  # create a new dict
    assert copied_dct is not exp_dict
    assert copied_dct == exp_dict

    # Convert from `dict` to `CCDCharacteristics`
    other_obj = CCDCharacteristics.from_dict(copied_dct)
    assert type(other_obj) == CCDCharacteristics
    assert obj == other_obj
    assert obj is not other_obj
    assert copied_dct == exp_dict

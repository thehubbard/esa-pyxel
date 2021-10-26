#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from copy import deepcopy

import pytest

from pyxel.detectors import Characteristics, CMOSCharacteristics


@pytest.mark.parametrize(
    "qe, eta, sv, amp, a1, a2, fwc, vg, dt, cutoff, vbiaspower, dsub, vreset, biasgate, preampref",
    [
        (0, 0, 0, 0, 0, 0, 0, 0.0, 0, 1.7, 0.0, 0.3, 0.0, 1.8, 0.0),
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
            15.0,
            3.4,
            1.0,
            0.3,
            2.6,
            4.0,
        ),
    ],
)
def test_create_valid_cmoscharacteristics(
    qe,
    eta,
    sv,
    amp,
    a1,
    a2,
    fwc,
    vg,
    dt,
    cutoff,
    vbiaspower,
    dsub,
    vreset,
    biasgate,
    preampref,
):
    """Test when creating a valid `CMOSCharacteristics` object."""
    _ = CMOSCharacteristics(
        qe=qe,
        eta=eta,
        sv=sv,
        amp=amp,
        a1=a1,
        a2=a2,
        fwc=fwc,
        vg=vg,
        dt=dt,
        cutoff=cutoff,
        vbiaspower=vbiaspower,
        dsub=dsub,
        vreset=vreset,
        biasgate=biasgate,
        preampref=preampref,
    )


@pytest.mark.parametrize(
    "other_obj, is_equal",
    [
        pytest.param(None, False, id="None"),
        pytest.param(CMOSCharacteristics(), False, id="Empty 'CMOSCharacteristics'"),
        pytest.param(CMOSCharacteristics(qe=0.1), False, id="Only one parameter"),
        pytest.param(
            Characteristics(
                qe=0.1, eta=0.2, sv=3.3, amp=4.4, a1=5, a2=6, fwc=7, vg=0.8, dt=9.9
            ),
            False,
            id="Wrong type",
        ),
        pytest.param(
            CMOSCharacteristics(
                qe=0.1,
                eta=0.2,
                sv=3.3,
                amp=4.4,
                a1=5,
                a2=6,
                fwc=7,
                vg=0.8,
                dt=9.9,
                cutoff=10.0,
                vbiaspower=1.0,
                dsub=0.5,
                vreset=0.2,
                biasgate=2.0,
                preampref=3.0,
            ),
            True,
            id="Almost same parameters, different class",
        ),
    ],
)
def test_is_equal(other_obj, is_equal):
    """Test equality statement for `CMOSCharacteristics`."""
    obj = CMOSCharacteristics(
        qe=0.1,
        eta=0.2,
        sv=3.3,
        amp=4.4,
        a1=5,
        a2=6,
        fwc=7,
        vg=0.8,
        dt=9.9,
        cutoff=10.0,
        vbiaspower=1.0,
        dsub=0.5,
        vreset=0.2,
        biasgate=2.0,
        preampref=3.0,
    )

    if is_equal:
        assert obj == other_obj
    else:
        assert obj != other_obj


@pytest.mark.parametrize(
    "obj, exp_dict",
    [
        (
            CMOSCharacteristics(
                qe=0.1,
                eta=0.2,
                sv=3.3,
                amp=4.4,
                a1=5,
                a2=6,
                fwc=7,
                vg=0.8,
                dt=9.9,
                cutoff=10.0,
                vbiaspower=1.0,
                dsub=0.5,
                vreset=0.2,
                biasgate=2.0,
                preampref=3.0,
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
                "cutoff": 10.0,
                "vbiaspower": 1.0,
                "dsub": 0.5,
                "vreset": 0.2,
                "biasgate": 2.0,
                "preampref": 3.0,
            },
        )
    ],
)
def test_to_and_from_dict(obj, exp_dict):
    """Test methods 'to_dict', 'from_dict'."""
    assert type(obj) == CMOSCharacteristics

    # Convert from `CMOSCharacteristics` to a `dict`
    dct = obj.to_dict()
    assert dct == exp_dict

    # Copy 'exp_dict'
    copied_dct = deepcopy(exp_dict)  # create a new dict
    assert copied_dct is not exp_dict
    assert copied_dct == exp_dict

    # Convert from `dict` to `CMOSCharacteristics`
    other_obj = CMOSCharacteristics.from_dict(copied_dct)
    assert type(other_obj) == CMOSCharacteristics
    assert obj == other_obj
    assert obj is not other_obj
    assert copied_dct == exp_dict

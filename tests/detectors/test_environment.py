#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from copy import deepcopy

import pytest

from pyxel.detectors import Environment


@pytest.mark.parametrize(
    "temperature, total_ionising_dose, total_non_ionising_dose",
    [(0.0, 0.0, 0.0), (1000.0, 1e15, 1e15)],
)
def test_create_environment(
    temperature: float, total_ionising_dose: float, total_non_ionising_dose: float
):
    """Test when creating a valid `Environment` object."""
    _ = Environment(
        temperature=temperature,
        total_ionising_dose=total_ionising_dose,
        total_non_ionising_dose=total_non_ionising_dose,
    )


@pytest.mark.parametrize(
    "temperature, total_ionising_dose, total_non_ionising_dose, exp_exc, exp_error",
    [
        pytest.param(-0.1, 0.0, 0.0, ValueError, r"\'temperature\'"),
        pytest.param(1001, 0.0, 0.0, ValueError, r"\'temperature\'"),
        pytest.param(0.0, -0.1, 0.0, ValueError, r"\'total_ionising_dose\'"),
        pytest.param(0.0, 1.1e15, 0.0, ValueError, r"\'total_ionising_dose\'"),
        pytest.param(0.0, 0.0, -0.1, ValueError, r"\'total_non_ionising_dose\'"),
        pytest.param(0.0, 0.0, 1.1e15, ValueError, r"\'total_non_ionising_dose\'"),
    ],
)
def test_create_invalid_environment(
    temperature: float,
    total_ionising_dose: float,
    total_non_ionising_dose: float,
    exp_exc,
    exp_error,
):
    """Test when creating an invalid `Environment` object."""
    with pytest.raises(exp_exc, match=exp_error):
        _ = Environment(
            temperature=temperature,
            total_ionising_dose=total_ionising_dose,
            total_non_ionising_dose=total_non_ionising_dose,
        )


@pytest.mark.parametrize(
    "other_obj, is_equal",
    [
        pytest.param(None, False, id="None"),
        pytest.param(Environment(), False, id="Empty 'Environment'"),
        pytest.param(Environment(temperature=100.1), False, id="Only one parameter"),
        pytest.param(
            Environment(
                temperature=100.1,
                total_ionising_dose=200.2,
                total_non_ionising_dose=300.3,
            ),
            True,
            id="Same parameters, same class",
        ),
    ],
)
def test_is_equal(other_obj, is_equal):
    """Test equality statement for Environment."""
    obj = Environment(
        temperature=100.1, total_ionising_dose=200.2, total_non_ionising_dose=300.3
    )

    if is_equal:
        assert obj == other_obj
    else:
        assert obj != other_obj


@pytest.mark.parametrize(
    "obj, exp_dict",
    [
        (
            Environment(
                temperature=100.1,
                total_ionising_dose=200.2,
                total_non_ionising_dose=300.3,
            ),
            {
                "temperature": 100.1,
                "total_ionising_dose": 200.2,
                "total_non_ionising_dose": 300.3,
            },
        )
    ],
)
def test_to_and_from_dict(obj, exp_dict):
    """Test methods 'to_dict' and 'from_dict'"""

    assert type(obj) == Environment

    # Convert from `Environment` to a `dict`
    dct = obj.to_dict()
    assert dct == exp_dict

    # Copy 'exp_dict'
    copied_dct = deepcopy(exp_dict)  # create a new dict
    assert copied_dct is not exp_dict
    assert copied_dct == exp_dict

    # Convert from `dict` to `Environment`
    other_obj = Environment.from_dict(copied_dct)
    assert type(other_obj) == Environment
    assert obj == other_obj
    assert obj is not other_obj
    assert copied_dct == exp_dict

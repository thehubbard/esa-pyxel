#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import pytest

from pyxel.observation.parameter_values import ParameterType, convert_values


@pytest.mark.parametrize(
    "values, parameter_type, exp_values",
    [
        pytest.param("_", ParameterType.Simple, "_", id="_ + simple"),
        pytest.param(
            ["_", "_"], ParameterType.Simple, ["_", "_"], id="list['_'] + simple"
        ),
        pytest.param(
            [1, 2, 3], ParameterType.Simple, [1, 2, 3], id="list[int] + simple"
        ),
        pytest.param(
            ["foo", "bar"],
            ParameterType.Simple,
            ["foo", "bar"],
            id="list[str] + simple",
        ),
        pytest.param(
            [[1, 2], [3, 4]],
            ParameterType.Simple,
            [[1, 2], [3, 4]],
            id="list[list[int]] + simple",
        ),
        pytest.param(
            ((1, 2), (3, 4)),
            ParameterType.Simple,
            ((1, 2), (3, 4)),
            id="tuple[tuple[int]] + simple",
        ),
        pytest.param(
            [[1, 2], [3, 4]],
            ParameterType.Multi,
            [(1, 2), (3, 4)],
            id="list[list[int]] + multi",
        ),
        pytest.param(
            [
                [[1, 0.5], [0.5, 1], [0, 0.5], [0, 0]],
                [[1, 0.3], [0.3, 1], [0, 0.3], [0, 0]],
            ],
            ParameterType.Multi,
            [
                (
                    (1, 0.5),
                    (0.5, 1),
                    (0, 0.5),
                    (
                        0,
                        0,
                    ),
                ),
                (
                    (1, 0.3),
                    (0.3, 1),
                    (0, 0.3),
                    (
                        0,
                        0,
                    ),
                ),
            ],
            id="3d array + multi",
        ),
    ],
)
def test_convert_values(values, parameter_type, exp_values):
    """Test function 'convert_values'."""
    result = convert_values(values=values, parameter_type=parameter_type)

    assert result == exp_values

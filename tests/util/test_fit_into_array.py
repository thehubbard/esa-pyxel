#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.


from typing import Literal, Optional

import numpy as np
import pytest

from pyxel.util import fit_into_array
from pyxel.util.image import Alignment, _set_relative_position


@pytest.fixture
def array_2d() -> np.ndarray:
    """Create 2D data."""
    array = np.array([[1, 2], [3, 4], [5, 6]])

    return array


@pytest.fixture
def array_1d_col() -> np.ndarray:
    """Create 2D data."""
    array = np.array([[1], [2], [3], [4], [5]])

    return array


@pytest.fixture
def array_1d_row() -> np.ndarray:
    """Create 2D data."""
    array = np.array([[1, 2, 3, 4, 5]])

    return array


@pytest.mark.parametrize(
    "output_shape, relative_position, align, allow_smaller_array, expected_output",
    [
        pytest.param(
            (5, 5),
            (0, 0),
            None,
            True,
            np.array(
                [
                    [1.0, 2.0, 0.0, 0.0, 0.0],
                    [3.0, 4.0, 0.0, 0.0, 0.0],
                    [5.0, 6.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
        pytest.param(
            (5, 5),
            (-1, -1),
            None,
            True,
            np.array(
                [
                    [4.0, 0.0, 0.0, 0.0, 0.0],
                    [6.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
        pytest.param(
            (5, 5),
            (3, 3),
            None,
            True,
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 2.0],
                    [0.0, 0.0, 0.0, 3.0, 4.0],
                ]
            ),
        ),
        pytest.param(
            (5, 5),
            (0, 0),
            "center",
            True,
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 2.0, 0.0, 0.0],
                    [0.0, 3.0, 4.0, 0.0, 0.0],
                    [0.0, 5.0, 6.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
        pytest.param(
            (5, 5),
            (0, 0),
            "bottom_left",
            True,
            np.array(
                [
                    [1.0, 2.0, 0.0, 0.0, 0.0],
                    [3.0, 4.0, 0.0, 0.0, 0.0],
                    [5.0, 6.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
        pytest.param(
            (5, 5),
            (0, 0),
            "top_left",
            True,
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 2.0, 0.0, 0.0, 0.0],
                    [3.0, 4.0, 0.0, 0.0, 0.0],
                    [5.0, 6.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
        pytest.param(
            (5, 5),
            (0, 0),
            "top_right",
            True,
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 2.0],
                    [0.0, 0.0, 0.0, 3.0, 4.0],
                    [0.0, 0.0, 0.0, 5.0, 6.0],
                ]
            ),
        ),
        pytest.param(
            (5, 5),
            (0, 0),
            "bottom_right",
            True,
            np.array(
                [
                    [0.0, 0.0, 0.0, 1.0, 2.0],
                    [0.0, 0.0, 0.0, 3.0, 4.0],
                    [0.0, 0.0, 0.0, 5.0, 6.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
    ],
)
def test_fit_into_array_2d(
    array_2d,
    output_shape: tuple[int, int],
    relative_position: tuple[int, int],
    align: Optional[
        Literal["center", "top_left", "top_right", "bottom_left", "bottom_right"]
    ],
    allow_smaller_array: bool,
    expected_output: np.ndarray,
):
    """Test function 'fit_into_array' with some valid inputs."""

    output = fit_into_array(
        array=array_2d,
        output_shape=output_shape,
        relative_position=relative_position,
        align=align,
        allow_smaller_array=allow_smaller_array,
    )

    np.testing.assert_array_almost_equal(output, expected_output)


@pytest.mark.parametrize(
    "output_shape, relative_position, align, allow_smaller_array, expected_output",
    [
        pytest.param(
            (5, 1),
            (0, 0),
            None,
            True,
            np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]),
        ),
        pytest.param(
            (5, 3),
            (0, 0),
            None,
            True,
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ]
            ),
        ),
        pytest.param(
            (5, 3),
            (0, 1),
            None,
            True,
            np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [0.0, 4.0, 0.0],
                    [0.0, 5.0, 0.0],
                ]
            ),
        ),
    ],
)
def test_fit_into_array_1d_col(
    array_1d_col,
    output_shape: tuple[int, int],
    relative_position: tuple[int, int],
    align: Optional[
        Literal["center", "top_left", "top_right", "bottom_left", "bottom_right"]
    ],
    allow_smaller_array: bool,
    expected_output: np.ndarray,
):
    """Test function 'fit_into_array' with some valid inputs."""

    output = fit_into_array(
        array=array_1d_col,
        output_shape=output_shape,
        relative_position=relative_position,
        align=align,
        allow_smaller_array=allow_smaller_array,
    )

    np.testing.assert_array_almost_equal(output, expected_output)


@pytest.mark.parametrize(
    "output_shape, relative_position, align, allow_smaller_array, expected_output",
    [
        pytest.param(
            (1, 5),
            (0, 0),
            None,
            True,
            np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        ),
        pytest.param(
            (5, 3),
            (4, 0),
            None,
            True,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0],
                ]
            ),
        ),
        pytest.param(
            (3, 5),
            (0, 0),
            None,
            True,
            np.array(
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
    ],
)
def test_fit_into_array_1d_row(
    array_1d_row,
    output_shape: tuple[int, int],
    relative_position: tuple[int, int],
    align: Optional[
        Literal["center", "top_left", "top_right", "bottom_left", "bottom_right"]
    ],
    allow_smaller_array: bool,
    expected_output: np.ndarray,
):
    """Test function 'fit_into_array' with some valid inputs."""

    output = fit_into_array(
        array=array_1d_row,
        output_shape=output_shape,
        relative_position=relative_position,
        align=align,
        allow_smaller_array=allow_smaller_array,
    )

    np.testing.assert_array_almost_equal(output, expected_output)


@pytest.mark.parametrize(
    "output_shape, relative_position, align, allow_smaller_array, exp_exc, exp_error",
    [
        pytest.param(
            (10, 10),
            (0, 0),
            "foo",
            True,
            ValueError,
            r"'.*' is not a valid Alignment",
        ),
        pytest.param(
            (20, 20),
            (0, 0),
            None,
            False,
            ValueError,
            "Input array too small to fit into the desired shape!.",
        ),
        pytest.param(
            (10, 10),
            (15, 15),
            None,
            True,
            ValueError,
            "No overlap of array and target in Y and X dimension.",
        ),
        pytest.param(
            (10, 10),
            (15, 0),
            None,
            True,
            ValueError,
            "No overlap of array and target in Y dimension.",
        ),
        pytest.param(
            (10, 10),
            (0, 15),
            None,
            True,
            ValueError,
            "No overlap of array and target in X dimension.",
        ),
    ],
)
def test_fit_into_array_bad_inputs(
    output_shape: tuple[int, int],
    relative_position: tuple[int, int],
    align: Optional[
        Literal["center", "top_left", "top_right", "bottom_left", "bottom_right"]
    ],
    allow_smaller_array: bool,
    exp_exc,
    exp_error,
):
    """Test function 'fit_into_array' with bad inputs."""

    array = np.ones((10, 10))

    with pytest.raises(exp_exc, match=exp_error):
        fit_into_array(
            array=array,
            output_shape=output_shape,
            relative_position=relative_position,
            align=align,
            allow_smaller_array=allow_smaller_array,
        )


@pytest.mark.parametrize(
    "output,array,alignment,expected",
    [
        ((None, None), (None, None), Alignment.bottom_left, (0, 0)),
        ((2, 3), (0, 0), Alignment.top_left, (2, 0)),
        ((2, 3), (1, 1), Alignment.top_left, (1, 0)),
        ((2, 3), (0, 0), Alignment.bottom_right, (0, 3)),
        ((2, 3), (1, 1), Alignment.bottom_right, (0, 2)),
        ((2, 3), (0, 0), Alignment.top_right, (2, 3)),
        ((2, 3), (1, 1), Alignment.top_right, (1, 2)),
        ((4, 6), (0, 0), Alignment.center, (2, 3)),
        ((4, 6), (2, 2), Alignment.center, (1, 2)),
        ((3, 5), (0, 0), Alignment.center, (1, 2)),
        ((3, 5), (1, 1), Alignment.center, (1, 2)),
        ((3, 5), (2, 2), Alignment.center, (0, 1)),
    ],
)
def test_set_relative_position(output, array, alignment, expected):
    """Test function 'pyxel.util.image._set_relative_position."""
    array_y, array_x = array
    output_y, output_x = output

    result = _set_relative_position(
        array_x=array_x,
        array_y=array_y,
        output_x=output_x,
        output_y=output_y,
        alignment=alignment,
    )
    assert result == expected

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import typing as t

import numpy as np
import pytest

from pyxel.models.optics.alignment import apply_alignment


@pytest.mark.parametrize("target_shape", [(0, 0), (1, 1)])
def test_apply_alignment_0x0(target_shape: t.Tuple[int, int]):
    """Test function 'apply_alignment' with a 2x4 array."""
    data_2d = np.array([]).reshape((0, 0))

    target_rows, target_cols = target_shape

    with pytest.raises(ValueError):
        _ = apply_alignment(
            data_2d=data_2d,
            target_rows=target_rows,
            target_cols=target_cols,
        )


@pytest.mark.parametrize(
    "target_shape, exp_data_2d",
    [
        pytest.param((2, 4), np.array([[0, 1, 2, 3], [4, 5, 6, 7]]), id="2x4"),
        pytest.param((2, 3), np.array([[0, 1, 2], [4, 5, 6]]), id="2x3"),
        pytest.param((2, 2), np.array([[1, 2], [5, 6]]), id="2x2"),
        pytest.param((2, 1), np.array([[1], [5]]), id="2x1"),
        pytest.param((1, 4), np.array([[0, 1, 2, 3]]), id="1x4"),
        pytest.param(
            (2, 5),
            None,
            marks=pytest.mark.xfail(raises=ValueError, strict=True),
            id="Too big - 2x5",
        ),
        pytest.param(
            (3, 4),
            None,
            marks=pytest.mark.xfail(raises=ValueError, strict=True),
            id="Too big - 3x4",
        ),
        pytest.param(
            (2, 0),
            None,
            marks=pytest.mark.xfail(raises=ValueError, strict=True),
            id="2x0",
        ),
        pytest.param(
            (0, 4),
            None,
            marks=pytest.mark.xfail(raises=ValueError, strict=True),
            id="0x4",
        ),
    ],
)
def test_apply_alignment_2x4(target_shape: t.Tuple[int, int], exp_data_2d: np.ndarray):
    """Test function 'apply_alignment' with a 2x4 array."""
    data_2d = np.arange(2 * 4).reshape((2, 4))

    target_rows, target_cols = target_shape

    result_data_2d = apply_alignment(
        data_2d=data_2d,
        target_rows=target_rows,
        target_cols=target_cols,
    )

    np.testing.assert_equal(result_data_2d, exp_data_2d)
    assert result_data_2d.shape == target_shape


@pytest.mark.parametrize(
    "target_shape, exp_data_2d",
    [
        pytest.param((3, 1), np.array([[0], [1], [2]]), id="3x1"),
    ],
)
def test_apply_alignment_3x1(target_shape: t.Tuple[int, int], exp_data_2d: np.ndarray):
    """Test function 'apply_alignment' with a 2x4 array."""
    data_2d = np.arange(3 * 1).reshape((3, 1))

    target_rows, target_cols = target_shape

    result_data_2d = apply_alignment(
        data_2d=data_2d,
        target_rows=target_rows,
        target_cols=target_cols,
    )

    np.testing.assert_equal(result_data_2d, exp_data_2d)
    assert result_data_2d.shape == target_shape


@pytest.mark.parametrize(
    "target_shape, exp_data_2d",
    [
        pytest.param((2, 4), np.array([[19, 20, 21, 22], [25, 26, 27, 28]]), id="2x4"),
        pytest.param(
            (3, 5),
            np.array(
                [[12, 13, 14, 15, 16], [18, 19, 20, 21, 22], [24, 25, 26, 27, 28]]
            ),
            id="3x5",
        ),
    ],
)
def test_apply_alignment_8x6(target_shape: t.Tuple[int, int], exp_data_2d: np.ndarray):
    """Test function 'apply_alignment' with a 8x6 array."""
    data_2d = np.arange(8 * 6).reshape((8, 6))

    target_rows, target_cols = target_shape

    result_data_2d = apply_alignment(
        data_2d=data_2d,
        target_rows=target_rows,
        target_cols=target_cols,
    )

    np.testing.assert_equal(result_data_2d, exp_data_2d)
    assert result_data_2d.shape == target_shape

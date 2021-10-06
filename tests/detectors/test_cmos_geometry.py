#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from copy import deepcopy

import pytest

from pyxel.detectors import CMOSGeometry, Geometry


@pytest.mark.parametrize("row", [0, 10000])
@pytest.mark.parametrize("col", [0, 10000])
@pytest.mark.parametrize("total_thickness", [0.0, 10000.0])
@pytest.mark.parametrize("pixel_vert_size", [0.0, 1000.0])
@pytest.mark.parametrize("pixel_horz_size", [0.0, 1000.0])
@pytest.mark.parametrize("n_output", [0, 32])
@pytest.mark.parametrize("n_row_overhead", [0, 100])
@pytest.mark.parametrize("n_frame_overhead", [0, 100])
@pytest.mark.parametrize("reverse_scan_direction", [True, False])
@pytest.mark.parametrize("reference_pixel_border_width", [0, 32])
def test_create_valid_geometry(
    row,
    col,
    total_thickness,
    pixel_vert_size,
    pixel_horz_size,
    n_output,
    n_row_overhead,
    n_frame_overhead,
    reverse_scan_direction,
    reference_pixel_border_width,
):
    """Test when creating a valid `Geometry` object."""
    _ = CMOSGeometry(
        row=row,
        col=col,
        total_thickness=total_thickness,
        pixel_vert_size=pixel_vert_size,
        pixel_horz_size=pixel_horz_size,
        n_output=n_output,
        n_row_overhead=n_row_overhead,
        n_frame_overhead=n_frame_overhead,
        reverse_scan_direction=reverse_scan_direction,
        reference_pixel_border_width=reference_pixel_border_width,
    )


@pytest.mark.parametrize(
    "row, col, total_thickness, pixel_vert_size, pixel_horz_size, "
    "n_output, n_row_overhead, n_frame_overhead, reverse_scan_direction, "
    "reference_pixel_border_width, exp_exc",
    [
        pytest.param(
            -1, 100, 100.0, 100.0, 100.0, 1, 0, 0, False, 4, ValueError, id="row < 0"
        ),
        pytest.param(
            10001,
            100,
            100.0,
            100.0,
            100.0,
            1,
            0,
            0,
            False,
            4,
            ValueError,
            id="row > 10000",
        ),
        pytest.param(
            100, -1, 100.0, 100.0, 100.0, 1, 0, 0, False, 4, ValueError, id="col < 0"
        ),
        pytest.param(
            100,
            10001,
            100.0,
            100.0,
            100.0,
            1,
            0,
            0,
            False,
            4,
            ValueError,
            id="col > 10000",
        ),
        pytest.param(
            100,
            100,
            -0.1,
            100.0,
            100.0,
            1,
            0,
            0,
            False,
            4,
            ValueError,
            id="total_thickness < 0.",
        ),
        pytest.param(
            100,
            100,
            10000.1,
            100.0,
            100.0,
            1,
            0,
            0,
            False,
            4,
            ValueError,
            id="total_thickness > 10000.",
        ),
        pytest.param(
            100,
            100,
            10000,
            100.0,
            100.0,
            -1,
            0,
            0,
            False,
            4,
            ValueError,
            id="n_output < 0",
        ),
        pytest.param(
            100,
            100,
            10000,
            100.0,
            100.0,
            33,
            0,
            0,
            False,
            4,
            ValueError,
            id="n_output > 32",
        ),
        pytest.param(
            100,
            100,
            10000,
            100.0,
            100.0,
            1,
            -1,
            0,
            False,
            4,
            ValueError,
            id="n_row_overhead < 0",
        ),
        pytest.param(
            100,
            100,
            10000,
            100.0,
            100.0,
            1,
            101,
            0,
            False,
            4,
            ValueError,
            id="n_row_overhead > 100",
        ),
        pytest.param(
            100,
            100,
            10000,
            100.0,
            100.0,
            1,
            0,
            -1,
            False,
            4,
            ValueError,
            id="n_frame_overhead < 0",
        ),
        pytest.param(
            100,
            100,
            10000,
            100.0,
            100.0,
            1,
            0,
            101,
            False,
            4,
            ValueError,
            id="n_frame_overhead > 100",
        ),
        pytest.param(
            100,
            100,
            10000,
            100.0,
            100.0,
            1,
            0,
            1,
            False,
            -1,
            ValueError,
            id="reference_pixel_border_width < 0",
        ),
        pytest.param(
            100,
            100,
            10000,
            100.0,
            100.0,
            1,
            0,
            1,
            False,
            33,
            ValueError,
            id="reference_pixel_border_width > 33",
        ),
    ],
)
def test_create_invalid_geometry(
    row,
    col,
    total_thickness,
    pixel_vert_size,
    pixel_horz_size,
    n_output,
    n_row_overhead,
    n_frame_overhead,
    reverse_scan_direction,
    reference_pixel_border_width,
    exp_exc,
):
    """Test when creating an invalid `Geometry` object."""
    with pytest.raises(exp_exc):
        _ = CMOSGeometry(
            row=row,
            col=col,
            total_thickness=total_thickness,
            pixel_vert_size=pixel_vert_size,
            pixel_horz_size=pixel_horz_size,
            n_output=n_output,
            n_row_overhead=n_row_overhead,
            n_frame_overhead=n_frame_overhead,
            reverse_scan_direction=reverse_scan_direction,
            reference_pixel_border_width=reference_pixel_border_width,
        )


@pytest.mark.parametrize(
    "other_obj, is_equal",
    [
        pytest.param(None, False, id="None"),
        pytest.param(CMOSGeometry(), False, id="Empty 'Geometry'"),
        pytest.param(CMOSGeometry(row=100), False, id="Only one parameter"),
        pytest.param(
            Geometry(
                row=100,
                col=120,
                total_thickness=123.1,
                pixel_horz_size=12.4,
                pixel_vert_size=34.5,
            ),
            False,
            id="Almost same parameters, different class",
        ),
        pytest.param(
            CMOSGeometry(
                row=100,
                col=120,
                total_thickness=123.1,
                pixel_horz_size=12.4,
                pixel_vert_size=34.5,
                n_output=2,
                n_row_overhead=13,
                n_frame_overhead=14,
                reverse_scan_direction=True,
                reference_pixel_border_width=8,
            ),
            True,
            id="Same parameters, same class",
        ),
    ],
)
def test_is_equal(other_obj, is_equal):
    """Test equality statement for Geometry."""
    obj = CMOSGeometry(
        row=100,
        col=120,
        total_thickness=123.1,
        pixel_horz_size=12.4,
        pixel_vert_size=34.5,
        n_output=2,
        n_row_overhead=13,
        n_frame_overhead=14,
        reverse_scan_direction=True,
        reference_pixel_border_width=8,
    )

    if is_equal:
        assert obj == other_obj
    else:
        assert obj != other_obj


@pytest.mark.parametrize(
    "obj, exp_dict",
    [
        (
            CMOSGeometry(
                row=100,
                col=120,
                total_thickness=123.1,
                pixel_horz_size=12.4,
                pixel_vert_size=34.5,
                n_output=2,
                n_row_overhead=10,
                n_frame_overhead=12,
                reverse_scan_direction=True,
                reference_pixel_border_width=5,
            ),
            {
                "row": 100,
                "col": 120,
                "total_thickness": 123.1,
                "pixel_horz_size": 12.4,
                "pixel_vert_size": 34.5,
                "n_output": 2,
                "n_row_overhead": 10,
                "n_frame_overhead": 12,
                "reverse_scan_direction": True,
                "reference_pixel_border_width": 5,
            },
        ),
    ],
)
def test_to_and_from_dict(obj, exp_dict):
    """Test methods 'to_dict', 'from_dict'."""
    assert type(obj) == CMOSGeometry

    # Convert from `CMOSGeometry` to a `dict`
    dct = obj.to_dict()
    assert dct == exp_dict

    # Copy 'exp_dict'
    copied_dct = deepcopy(exp_dict)  # create a new dict
    assert copied_dct is not exp_dict
    assert copied_dct == exp_dict

    # Convert from `dict` to `CMOSGeometry`
    other_obj = CMOSGeometry.from_dict(copied_dct)
    assert type(other_obj) == CMOSGeometry
    assert obj == other_obj
    assert obj is not other_obj
    assert copied_dct == exp_dict

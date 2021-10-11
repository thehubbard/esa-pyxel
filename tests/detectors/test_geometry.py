#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import typing as t
from dataclasses import dataclass

import numpy as np
import pytest

from pyxel.detectors import CCDGeometry, CMOSGeometry, Geometry


@dataclass
class Parameters:
    """Only for testing."""

    row: int
    col: int
    pixel_vert_size: float
    pixel_horz_size: float


@pytest.mark.parametrize("geometry_cls", [CCDGeometry, CMOSGeometry])
@pytest.mark.parametrize(
    "parameters, exp_values",
    [
        pytest.param(
            Parameters(row=3, col=2, pixel_vert_size=0.1, pixel_horz_size=0.2),
            np.array([0.05, 0.05, 0.15, 0.15, 0.25, 0.25]),
            id="3x2",
        ),
        pytest.param(
            Parameters(row=3, col=1, pixel_vert_size=0.1, pixel_horz_size=0.2),
            np.array([0.05, 0.15, 0.25]),
            id="3x1",
        ),
        pytest.param(
            Parameters(row=1, col=3, pixel_vert_size=0.1, pixel_horz_size=0.2),
            np.array([0.05, 0.05, 0.05]),
            id="1x3",
        ),
    ],
)
def test_vertical_pixel_center_pos(
    geometry_cls: t.Type[Geometry], parameters: Parameters, exp_values: np.ndarray
):
    """Test method '.vertical_pixel_center_pos_list'."""
    # Create the geometry object
    geometry = geometry_cls(
        row=parameters.row,
        col=parameters.col,
        pixel_vert_size=parameters.pixel_vert_size,
        pixel_horz_size=parameters.pixel_horz_size,
    )

    # Get the positions
    values = geometry.vertical_pixel_center_pos_list()

    # Check the positions
    np.testing.assert_allclose(values, exp_values)


@pytest.mark.parametrize("geometry_cls", [CCDGeometry, CMOSGeometry])
@pytest.mark.parametrize(
    "parameters, exp_values",
    [
        pytest.param(
            Parameters(row=3, col=2, pixel_vert_size=0.1, pixel_horz_size=0.2),
            np.array([0.1, 0.3, 0.1, 0.3, 0.1, 0.3]),
            id="3x2",
        ),
        pytest.param(
            Parameters(row=3, col=1, pixel_vert_size=0.1, pixel_horz_size=0.2),
            np.array([0.1, 0.1, 0.1]),
            id="3x1",
        ),
        pytest.param(
            Parameters(row=1, col=3, pixel_vert_size=0.1, pixel_horz_size=0.2),
            np.array([0.1, 0.3, 0.5]),
            id="1x3",
        ),
    ],
)
def test_horizontal_pixel_center_pos(
    geometry_cls: t.Type[Geometry], parameters: Parameters, exp_values: np.ndarray
):
    """Test method '.horizontal_pixel_center_pos_list'."""
    # Create the geometry object
    geometry = geometry_cls(
        row=parameters.row,
        col=parameters.col,
        pixel_vert_size=parameters.pixel_vert_size,
        pixel_horz_size=parameters.pixel_horz_size,
    )

    # Get the positions
    values = geometry.horizontal_pixel_center_pos_list()

    np.testing.assert_allclose(values, exp_values)

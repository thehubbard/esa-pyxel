#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
import typing as t
from pathlib import Path

import numpy as np
import pytest

from pyxel.detectors import (
    CCD,
    CCDCharacteristics,
    CCDGeometry,
    Environment,
    CMOS,
    CMOSCharacteristics,
    CMOSGeometry,
    ReadoutProperties,
)
from pyxel.models.charge_collection import persistence


@pytest.fixture
def ccd_5x5() -> CCD:
    """Create a valid CCD detector."""
    detector = CCD(
        geometry=CCDGeometry(
            row=5,
            col=5,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(),
        characteristics=CCDCharacteristics(),
    )
    detector._readout_properties = ReadoutProperties(num_steps=1)
    return detector


@pytest.fixture
def cmos_5x5() -> CMOS:
    """Create a valid CCD detector."""
    detector = CMOS(
        geometry=CMOSGeometry(
            row=5,
            col=5,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(),
        characteristics=CMOSCharacteristics(),
    )
    detector._readout_properties = ReadoutProperties(num_steps=1)
    return detector


@pytest.fixture
def valid_density_map_path(
    tmp_path: Path,
) -> str:
    """Create valid 2D file on a temporary folder."""
    data_2d = np.ones((5, 5)) * 0.01

    final_path = f"{tmp_path}/densities.npy"
    np.save(final_path, arr=data_2d)

    return final_path


@pytest.fixture
def valid_capacity_map_path(
    tmp_path: Path,
) -> str:
    """Create valid 2D file on a temporary folder."""
    data_2d = np.ones((5, 5)) * 100.0

    final_path = f"{tmp_path}/capacities.npy"
    np.save(final_path, arr=data_2d)

    return final_path


@pytest.fixture
def invalid_density_map_path(
    tmp_path: Path,
) -> str:
    """Create valid 2D file on a temporary folder."""

    data_2d = np.array(
        [
            [1.0, 1.0, 0.5, 1.0, 1.0],
            [1.0, 2.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 2.0, 1.0],
            [0.5, 1.0, 1.0, 1.0, 1.0],
        ]
    )

    final_path = f"{tmp_path}/invalid_densities.npy"
    np.save(final_path, arr=data_2d)

    return final_path


@pytest.mark.parametrize(
    "trap_time_constants, trap_proportions",
    [
        pytest.param([1.0, 10.0], [0.4, 0.6], id="valid"),
    ],
)
def test_persistence_valid(
    cmos_5x5: CMOS,
    trap_time_constants: t.Sequence[float],
    trap_proportions: t.Sequence[float],
    valid_density_map_path: str,
    valid_capacity_map_path: str,
):

    detector = cmos_5x5

    assert not detector.has_persistence()

    persistence(
        detector=detector,
        trap_time_constants=trap_time_constants,
        trap_proportions=trap_proportions,
        trap_densities_filename=valid_density_map_path,
        trap_capacities_filename=valid_capacity_map_path,
    )

    assert detector.has_persistence()
    assert len(detector.persistence.trap_list) == len(trap_time_constants)


@pytest.mark.parametrize(
    "trap_time_constants, trap_proportions, exp_error, exp_msg",
    [
        pytest.param(
            [],
            [],
            ValueError,
            "Expecting at least one 'trap_time_constants' and 'trap_proportions'",
            id="no elements",
        ),
        pytest.param(
            [1.0],
            [0.1, 0.1],
            ValueError,
            "Expecting same number of elements for parameters",
            id="not same number of elements",
        ),
    ],
)
def test_persistence_invalid(
    cmos_5x5: CMOS,
    trap_time_constants: t.Sequence[float],
    trap_proportions: t.Sequence[float],
    valid_density_map_path: str,
    valid_capacity_map_path: str,
    exp_error,
    exp_msg,
):

    detector = cmos_5x5

    with pytest.raises(exp_error, match=exp_msg):
        persistence(
            detector=detector,
            trap_time_constants=trap_time_constants,
            trap_proportions=trap_proportions,
            trap_densities_filename=valid_density_map_path,
            trap_capacities_filename=valid_capacity_map_path,
        )


def test_persistence_with_ccd(
    ccd_5x5: CCD, valid_density_map_path: str, valid_capacity_map_path: str
):
    """Test model 'persistence' with a `CCD` detector."""
    with pytest.raises(TypeError, match="Expecting a CMOS object for detector."):
        detector = ccd_5x5

        persistence(
            detector=detector,
            trap_time_constants=[1.0, 10.0],
            trap_proportions=[0.4, 0.6],
            trap_densities_filename=valid_density_map_path,
            trap_capacities_filename=valid_capacity_map_path,
        )


def test_persistence_with_invalid_density_map(
    cmos_5x5: CMOS, invalid_density_map_path: str
):
    """Test model 'persistence' with an invalid density map."""
    with pytest.raises(ValueError, match="Trap density map values not between 0 and 1."):
        detector = cmos_5x5

        persistence(
            detector=detector,
            trap_time_constants=[1.0, 10.0],
            trap_proportions=[0.4, 0.6],
            trap_densities_filename=invalid_density_map_path,
        )


# @pytest.mark.parametrize(
#     "qe, exp_exc, exp_error",
#     [
#         pytest.param(1.5, ValueError, "Quantum efficiency not between 0 and 1."),
#     ],
# )
# def test_simple_conversion_valid(
#     ccd_5x5: CCD,
#     qe: float,
#     exp_exc,
#     exp_error,
# ):
#
#     with pytest.raises(exp_exc, match=exp_error):
#         simple_conversion(detector=ccd_5x5, quantum_efficiency=qe)
#
#
# def test_conversion_with_qe_valid(ccd_5x5: CCD, valid_qe_map_path: t.Union[str, Path]):
#
#     detector = ccd_5x5
#
#     array = np.ones((5, 5))
#     detector.photon.array = array
#     target = array * 0.5
#
#     conversion_with_qe_map(detector=detector, filename=valid_qe_map_path)
#
#     np.testing.assert_array_almost_equal(detector.charge.array, target)
#
#
# def test_simple_conversion_invalid(
#     ccd_5x5: CCD, invalid_qe_map_path: t.Union[str, Path]
# ):
#
#     with pytest.raises(
#         ValueError, match="Quantum efficiency values not between 0 and 1."
#     ):
#         conversion_with_qe_map(detector=ccd_5x5, filename=invalid_qe_map_path)

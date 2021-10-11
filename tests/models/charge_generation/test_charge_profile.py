#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.


from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyxel.detectors import (
    CCD,
    CCDCharacteristics,
    CCDGeometry,
    Detector,
    Environment,
    Material,
)
from pyxel.models.charge_generation import charge_profile


@pytest.fixture
def ccd_10x1() -> CCD:
    """Create a valid CCD detector."""
    return CCD(
        geometry=CCDGeometry(
            row=10,
            col=1,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        material=Material(),
        environment=Environment(),
        characteristics=CCDCharacteristics(),
    )


@pytest.fixture
def profile_10x1() -> np.ndarray:
    """Create a profile of 10x1."""
    data_1d = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    return data_1d


@pytest.fixture
def profile_10x1_txt_filename(profile_10x1: np.ndarray, tmp_path: Path) -> Path:
    """Create a filename with a profile of 10x1."""
    filename = tmp_path / "profile_10x1.txt"
    np.savetxt(fname=filename, X=profile_10x1)

    return filename


@pytest.fixture
def ccd_10x3() -> CCD:
    """Create a valid CCD detector."""
    return CCD(
        geometry=CCDGeometry(
            row=10,
            col=3,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        material=Material(),
        environment=Environment(),
        characteristics=CCDCharacteristics(),
    )


@pytest.fixture
def profile_10x3() -> np.ndarray:
    """Create a profile of 10x3."""
    data_2d = np.zeros(shape=(10, 3))
    data_2d[:5, 0] = 100.0
    data_2d[:5, 1] = 200.0
    data_2d[:5, 2] = 300.0

    data_2d = np.array(
        [
            [100.0, 200.0, 300.0],
            [100.0, 200.0, 300.0],
            [100.0, 200.0, 300.0],
            [100.0, 200.0, 300.0],
            [100.0, 200.0, 300.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    return data_2d


@pytest.fixture
def profile_10x3_txt_filename(profile_10x3: np.ndarray, tmp_path: Path) -> Path:
    """Create a filename with a profile of 10x3."""
    filename = tmp_path / "profile_10x3.txt"
    np.savetxt(fname=filename, X=profile_10x3)

    return filename


def test_charge_profile_10x1(ccd_10x1: CCD, profile_10x1_txt_filename: Path) -> None:
    """Test function 'charge_profile'."""
    detector = ccd_10x1  # type: Detector

    # Check initial detector object
    assert detector.geometry.row == 10
    assert detector.geometry.col == 1
    assert detector.charge.frame.empty

    # Run model
    charge_profile(
        detector=detector,
        txt_file=profile_10x1_txt_filename,
        fit_profile_to_det=True,
    )

    # Check modified charges in 'detector'
    exp_charges = pd.DataFrame(
        {
            "charge": [-1.0, -1.0, -1.0, -1.0, -1.0],
            "number": [100.0, 100.0, 100.0, 100.0, 100.0],
            "init_energy": [0.0, 0.0, 0.0, 0.0, 0.0],
            "energy": [0.0, 0.0, 0.0, 0.0, 0.0],
            "init_pos_ver": [5.0, 15.0, 25.0, 35.0, 45.0],
            "init_pos_hor": [5.0, 5.0, 5.0, 5.0, 5.0],
            "init_pos_z": [0.0, 0.0, 0.0, 0.0, 0.0],
            "position_ver": [5.0, 15.0, 25.0, 35.0, 45.0],
            "position_hor": [5.0, 5.0, 5.0, 5.0, 5.0],
            "position_z": [0.0, 0.0, 0.0, 0.0, 0.0],
            "velocity_ver": [0.0, 0.0, 0.0, 0.0, 0.0],
            "velocity_hor": [0.0, 0.0, 0.0, 0.0, 0.0],
            "velocity_z": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    charges = detector.charge.frame

    pd.testing.assert_frame_equal(charges, exp_charges)


def test_charge_profile_10x3(ccd_10x3: CCD, profile_10x3_txt_filename: Path) -> None:
    """Test function 'charge_profile'."""
    detector = ccd_10x3  # type: Detector

    # Check initial detector object
    assert detector.geometry.row == 10
    assert detector.geometry.col == 3
    assert detector.charge.frame.empty

    # Run model
    charge_profile(
        detector=detector,
        txt_file=profile_10x3_txt_filename,
        fit_profile_to_det=True,
    )

    # Check modified charges in 'detector'
    exp_charges = pd.DataFrame(
        {
            "charge": [-1.0, -1.0, -1.0] * 5,
            "number": [100.0, 200.0, 300.0] * 5,
            "init_energy": [0.0, 0.0, 0.0] * 5,
            "energy": [0.0, 0.0, 0.0] * 5,
            "init_pos_ver": [
                5.0,
                5.0,
                5.0,
                15.0,
                15.0,
                15.0,
                25.0,
                25.0,
                25.0,
                35.0,
                35.0,
                35.0,
                45.0,
                45.0,
                45.0,
            ],
            "init_pos_hor": [5.0, 15.0, 25.0] * 5,
            "init_pos_z": [0.0, 0.0, 0.0] * 5,
            "position_ver": [
                5.0,
                5.0,
                5.0,
                15.0,
                15.0,
                15.0,
                25.0,
                25.0,
                25.0,
                35.0,
                35.0,
                35.0,
                45.0,
                45.0,
                45.0,
            ],
            "position_hor": [5.0, 15.0, 25.0] * 5,
            "position_z": [0.0, 0.0, 0.0] * 5,
            "velocity_ver": [0.0, 0.0, 0.0] * 5,
            "velocity_hor": [0.0, 0.0, 0.0] * 5,
            "velocity_z": [0.0, 0.0, 0.0] * 5,
        }
    )
    charges = detector.charge.frame

    pd.testing.assert_frame_equal(charges, exp_charges)

from pathlib import Path
from typing import Union

import numpy as np
import pytest

from pyxel.detectors import CCD, CCDCharacteristics, CCDGeometry, Environment
from pyxel.models.charge_collection import fix_pattern_noise


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
    return detector


@pytest.fixture
def valid_noise_path(
    tmp_path: Path,
) -> str:
    """Create valid 2D file on a temporary folder."""
    data_2d = (
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        * 0.5
    )

    final_path = f"{tmp_path}/noise.npy"
    np.save(final_path, arr=data_2d)

    return final_path


def test_fix_pattern_noise_valid(ccd_5x5: CCD, valid_noise_path: Union[str, Path]):
    """Test function fix_pattern_noise with valid inputs."""

    detector = ccd_5x5

    array = np.ones((5, 5))
    detector.pixel.array = array
    target = array * 0.5

    fix_pattern_noise(detector=detector, filename=valid_noise_path)

    np.testing.assert_array_almost_equal(detector.pixel.array, target)

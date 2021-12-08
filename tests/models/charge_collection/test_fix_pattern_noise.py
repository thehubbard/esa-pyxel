#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from pyxel.detectors import CCD, CCDCharacteristics, CCDGeometry, Environment, Material
from pyxel.models.charge_collection import fix_pattern_noise


@pytest.fixture
def ccd_5x10() -> CCD:
    """Create a valid CCD detector."""
    return CCD(
        geometry=CCDGeometry(
            row=5,
            col=10,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        material=Material(),
        environment=Environment(),
        characteristics=CCDCharacteristics(),
    )


@pytest.fixture
def non_uniformity_2d() -> np.ndarray:
    data_2d = np.full(shape=(5, 10), fill_value=1.0)

    return data_2d


@pytest.fixture
def non_uniformity_fits(non_uniformity_2d: np.ndarray, tmp_path: Path) -> Path:
    filename = tmp_path / "non_uniformity.fits"

    fits.writeto(filename, data=non_uniformity_2d)

    return filename


@pytest.fixture
def non_uniformity_1d_npy(non_uniformity_2d: np.ndarray, tmp_path: Path) -> Path:
    filename = tmp_path / "non_uniformity.npy"

    data_1d = non_uniformity_2d.flatten()
    np.save(filename, arr=data_1d)

    return filename


def test_fix_pattern_noise_default(ccd_5x10: CCD):
    """Test model 'fix_pattern_noise' without a filename."""
    detector = ccd_5x10

    fix_pattern_noise(detector)


def test_fix_pattern_noise_with_2d_fits(ccd_5x10: CCD, non_uniformity_fits: Path):
    """Test model 'fix_pattern_noise' without a filename."""
    detector = ccd_5x10

    fix_pattern_noise(detector=detector, pixel_non_uniformity=str(non_uniformity_fits))


def test_fix_pattern_noise_with_1d_npy(ccd_5x10: CCD, non_uniformity_1d_npy: Path):
    """Test model 'fix_pattern_noise' without a filename."""
    detector = ccd_5x10

    fix_pattern_noise(
        detector=detector, pixel_non_uniformity=str(non_uniformity_1d_npy)
    )


def test_fix_pattern_noise_filename_not_found(ccd_5x10: CCD):
    """Test model 'fix_pattern_noise' with a filename not found."""
    detector = ccd_5x10

    with pytest.raises(FileNotFoundError, match="Cannot find filename"):
        fix_pattern_noise(detector=detector, pixel_non_uniformity="missing_file.fits")

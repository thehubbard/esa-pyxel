#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2021.
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
    Material,
    ReadoutProperties,
)
from pyxel.models.photon_generation import load_image


@pytest.fixture
def valid_data2d(
    tmp_path: Path,
) -> str:
    """Create valid 2D file on a temporary folder."""
    data_2d = np.ones((20, 20), dtype=np.uint16)
    final_path = f"{tmp_path}/img.npy"
    np.save(final_path, arr=data_2d)

    data_2d = np.ones((5, 5), dtype=np.uint16)
    final_path = f"{tmp_path}/img_invalid.npy"
    np.save(final_path, arr=data_2d)

    return tmp_path


@pytest.fixture
def ccd_10x10() -> CCD:
    """Create a valid CCD detector."""
    detector = CCD(
        geometry=CCDGeometry(
            row=10,
            col=10,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        material=Material(),
        environment=Environment(),
        characteristics=CCDCharacteristics(),
    )
    detector._readout_properties = ReadoutProperties(num_steps=1)
    return detector


@pytest.mark.parametrize(
    "image_file, fit_image_to_det, position, convert_to_photons, multiplier, time_scale",
    [
        pytest.param("img.npy", True, (0, 0), True, 1.0, 1.0, id="valid"),
        pytest.param(
            "img_invalid.npy",
            True,
            (0, 0),
            True,
            1.0,
            1.0,
            id="img_too_small",
            marks=pytest.mark.xfail(raises=ValueError, strict=True),
        ),
        pytest.param(
            "img.npy",
            True,
            (15, 15),
            True,
            1.0,
            1.0,
            id="out_of_bounds",
            marks=pytest.mark.xfail(raises=ValueError, strict=True),
        ),
        pytest.param(
            "img.npy",
            True,
            (15, 15),
            True,
            1.0,
            1.0,
            id="negative_position",
            marks=pytest.mark.xfail(raises=ValueError, strict=True),
        ),
    ],
)
def test_load_image(
    ccd_10x10: CCD,
    valid_data2d: str,
    image_file: str,
    fit_image_to_det: bool,
    position: t.Tuple[int, int],
    convert_to_photons: bool,
    multiplier: float,
    time_scale: float,
):
    """Test input parameters for function 'load_image'."""
    load_image(
        detector=ccd_10x10,
        image_file=f"{valid_data2d}/{image_file}",
        fit_image_to_det=fit_image_to_det,
        position=position,
        convert_to_photons=convert_to_photons,
        multiplier=multiplier,
        time_scale=time_scale,
    )

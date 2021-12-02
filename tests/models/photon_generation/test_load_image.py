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
    "image_file, position, align, convert_to_photons, multiplier, time_scale",
    [
        pytest.param("img.npy", (0, 0), None, True, 1.0, 1.0, id="valid"),
    ],
)
def test_load_image(
    ccd_10x10: CCD,
    valid_data2d: str,
    image_file: str,
    position: t.Tuple[int, int],
    align: t.Optional[
        t.Literal["center", "top_left", "top_right", "bottom_left", "bottom_right"]
    ],
    convert_to_photons: bool,
    multiplier: float,
    time_scale: float,
):
    """Test input parameters for function 'load_image'."""
    load_image(
        detector=ccd_10x10,
        image_file=f"{valid_data2d}/{image_file}",
        position=position,
        align=align,
        convert_to_photons=convert_to_photons,
        multiplier=multiplier,
        time_scale=time_scale,
    )

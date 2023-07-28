#   Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#  #
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this Pyxel package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.

import pytest
import xarray as xr
from datatree import DataTree

from pyxel.detectors import (
    CCD,
    CCDGeometry,
    Characteristics,
    Environment,
    ReadoutProperties,
)
from pyxel.models.data_processing import statistics


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
        environment=Environment(),
        characteristics=Characteristics(),
    )
    detector._readout_properties = ReadoutProperties(times=[1.0])
    return detector


def test_statistics(ccd_10x10: CCD):
    """Test input parameters for function 'statistics'."""
    detector = ccd_10x10
    statistics(detector=detector)

    data = detector.data
    assert isinstance(data, DataTree)

    for name in ["pixel", "photon", "signal", "image"]:
        data_statistics = data[f"/statistics/{name}"]
        assert isinstance(data_statistics, DataTree)

        assert "time" in data_statistics.coords
        assert list(data_statistics.coords["time"]) == [0.0]

        dataset = data_statistics.to_dataset()
        assert isinstance(dataset, xr.Dataset)

        assert "time" in dataset.coords
        assert list(dataset.coords["time"]) == [0.0]

    # Old tests
    # dataset = detector.processed_data.data
    # assert "time" in dataset.coords
    # assert list(dataset.coords["time"].values) == [0.0]

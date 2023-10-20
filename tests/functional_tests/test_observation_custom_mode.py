#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from pathlib import Path

import pytest

import pyxel


@pytest.fixture
def input_folder(request: pytest.FixtureRequest) -> Path:
    """Get a valid existing YAML filename."""
    filename: Path = request.path.parent
    return filename.resolve(strict=True)


def test_observation_custom_mode(input_folder: Path):
    """Test Observation custom mode with a bad custom file."""
    config_filename = input_folder / "data/observation_custom.yaml"
    assert config_filename.exists()

    with pytest.raises(ValueError, match=r"Custom data file has"):
        _ = pyxel.load(config_filename)

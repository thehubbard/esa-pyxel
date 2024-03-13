#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from pathlib import Path

import pytest

import pyxel


@pytest.fixture
def filename_exposure_no_outputs(request: pytest.FixtureRequest) -> Path:
    """Get a valid existing YAML filename."""
    filename: Path = request.path.parent / "data/simple_exposure_no_outputs.yaml"
    return filename.resolve(strict=True)


@pytest.fixture
def filename_observation_no_outputs(request: pytest.FixtureRequest) -> Path:
    """Get a valid existing YAML filename."""
    filename: Path = request.path.parent / "data/simple_observation_no_outputs.yaml"
    return filename.resolve(strict=True)


def test_exposure_no_outputs(filename_exposure_no_outputs: Path):
    """Test 'pyxel.run_mode' without outputs."""
    cfg = pyxel.load(filename_exposure_no_outputs)

    _ = pyxel.run_mode(
        mode=cfg.running_mode,
        detector=cfg.detector,
        pipeline=cfg.pipeline,
    )


def test_observation_no_outputs(filename_observation_no_outputs: Path):
    """Test 'pyxel.run_mode' without outputs."""
    cfg = pyxel.load(filename_observation_no_outputs)

    _ = pyxel.run_mode(
        mode=cfg.running_mode,
        detector=cfg.detector,
        pipeline=cfg.pipeline,
    )

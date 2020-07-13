from collections import abc
from pathlib import Path

import pytest

from pyxel import inputs_outputs as io
from pyxel.detectors import CCD
from pyxel.parametric.parametric import (
    Configuration,
    ParametricAnalysis,
    ParametricMode,
)
from pyxel.pipelines import DetectionPipeline, Processor

try:
    import pygmo as pg

    WITH_PYGMO = True
except ImportError:
    WITH_PYGMO = False


expected_sequential = [
    (0, [("level", 10), ("initial_energy", 100)]),
    (1, [("level", 20), ("initial_energy", 100)]),
    (2, [("level", 30), ("initial_energy", 100)]),
    (3, [("level", 100), ("initial_energy", 100)]),
    (4, [("level", 100), ("initial_energy", 200)]),
    (5, [("level", 100), ("initial_energy", 300)]),
]

expected_embedded = [
    (0, [("level", 10), ("initial_energy", 100)]),
    (1, [("level", 10), ("initial_energy", 200)]),
    (2, [("level", 10), ("initial_energy", 300)]),
    (3, [("level", 20), ("initial_energy", 100)]),
    (4, [("level", 20), ("initial_energy", 200)]),
    (5, [("level", 20), ("initial_energy", 300)]),
    (6, [("level", 30), ("initial_energy", 100)]),
    (7, [("level", 30), ("initial_energy", 200)]),
    (8, [("level", 30), ("initial_energy", 300)]),
]


@pytest.mark.skipif(not WITH_PYGMO, reason="Package 'pygmo' is not installed.")
@pytest.mark.parametrize(
    "mode, expected",
    [
        # ('single', expected_single),
        (ParametricMode.Sequential, expected_sequential),
        (ParametricMode.Embedded, expected_embedded),
    ],
)
def test_pipeline_parametric_without_init_photon(mode: ParametricMode, expected):
    input_filename = "tests/data/parametric.yaml"
    cfg = io.load(Path(input_filename))

    assert isinstance(cfg, dict)
    assert "simulation" in cfg
    assert "ccd_detector" in cfg
    assert "pipeline" in cfg

    simulation = cfg["simulation"]
    assert isinstance(simulation, Configuration)

    parametric = simulation.parametric
    assert isinstance(parametric, ParametricAnalysis)

    parametric.parametric_mode = mode

    detector = cfg["ccd_detector"]
    assert isinstance(detector, CCD)

    assert detector.has_photon is False
    with pytest.raises(
        RuntimeError,
        match=(
            r"Photon array is not initialized ! "
            r"Please use a 'Photon Generation' model"
        ),
    ):
        _ = detector.photon

    pipeline = cfg["pipeline"]
    assert isinstance(pipeline, DetectionPipeline)

    processor = Processor(
        detector=detector, pipeline=pipeline
    )  # type: pyxel.pipelines.processor.Processor
    result = parametric.debug(processor)
    assert result == expected

    configs = parametric.collect(processor)
    assert isinstance(configs, abc.Iterator)

    for config in configs:
        assert isinstance(config, Processor)

        config.run_pipeline()

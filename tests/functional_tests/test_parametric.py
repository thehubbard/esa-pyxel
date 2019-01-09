from pathlib import Path
import pytest
import pyxel    # noqa: F401
import esapy_config as om
from pyxel.pipelines.processor import Processor


expected_sequential = [
    (0, [('level', 10), ('initial_energy', 100)]),
    (1, [('level', 20), ('initial_energy', 100)]),
    (2, [('level', 30), ('initial_energy', 100)]),
    (3, [('level', 100), ('initial_energy', 100)]),
    (4, [('level', 100), ('initial_energy', 200)]),
    (5, [('level', 100), ('initial_energy', 300)])
]

expected_embedded = [
    (0, [('level', 10), ('initial_energy', 100)]),
    (1, [('level', 10), ('initial_energy', 200)]),
    (2, [('level', 10), ('initial_energy', 300)]),
    (3, [('level', 20), ('initial_energy', 100)]),
    (4, [('level', 20), ('initial_energy', 200)]),
    (5, [('level', 20), ('initial_energy', 300)]),
    (6, [('level', 30), ('initial_energy', 100)]),
    (7, [('level', 30), ('initial_energy', 200)]),
    (8, [('level', 30), ('initial_energy', 300)])
]


@pytest.mark.parametrize("mode, expected", [
    # ('single', expected_single),
    ('sequential', expected_sequential),
    ('embedded', expected_embedded),
])
def test_pipeline_parametric(mode, expected):
    input_filename = 'tests/data/pipeline_parametric.yaml'
    cfg = om.load(Path(input_filename))
    simulation = cfg.pop('simulation')
    parametric_analysis = simulation.parametric_analysis
    parametric_analysis.parametric_mode = mode
    detector = cfg['detector']
    pipeline = cfg['pipeline']
    processor = Processor(detector, pipeline)  # type: pyxel.pipelines.processor.Processor
    result = parametric_analysis.debug(processor)
    assert result == expected
    configs = parametric_analysis.collect(processor)
    for config in configs:
        # detector = config.pipeline.run_pipeline(config.detector)
        pass


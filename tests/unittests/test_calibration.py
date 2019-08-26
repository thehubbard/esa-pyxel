"""Unittests for the 'Calibration' class."""

import pytest

import pyxel.io as io
from pyxel.calibration.util import check_ranges, list_to_slice, read_data
from pyxel.detectors import CCD
from pyxel.parametric.parametric import Configuration
from pyxel.pipelines.pipeline import DetectionPipeline
from pyxel.pipelines.processor import Processor

try:
    import pygmo as pg
    WITH_PYGMO = True
except ImportError:
    WITH_PYGMO = False


@pytest.mark.skipif(not WITH_PYGMO, reason="Package 'pygmo' is not installed.")
@pytest.mark.parametrize('yaml',
                         [
                             'tests/data/calibrate.yaml',
                             'tests/data/calibrate_sade.yaml',
                             'tests/data/calibrate_sga.yaml',
                             'tests/data/calibrate_nlopt.yaml',
                             'tests/data/calibrate_models.yaml',
                             'tests/data/calibrate_custom_fitness.yaml',
                             'tests/data/calibrate_fits.yaml',
                         ])
def test_set_algo(yaml):
    """Test """
    cfg = io.load(yaml)
    simulation = cfg['simulation']
    obj = simulation.calibration.algorithm.get_algorithm()
    if isinstance(obj, pg.sade):
        pass
    elif isinstance(obj, pg.sga):
        pass
    elif isinstance(obj, pg.nlopt):
        pass
    else:
        raise ReferenceError


@pytest.mark.parametrize('input_data',
                         [
                          'tests/data/calibrate-data.npy',
                          'tests/data/calibrate-data.txt',
                          ['tests/data/calibrate-data.npy'],
                          ['tests/data/calibrate-data.txt'],
                          # 'tests/data/ascii_input_0.data',
                          'tests/data/ascii_input_1.data',
                          'tests/data/ascii_input_2.data',
                          ['tests/data/ascii_input_3.data'],
                          ['tests/data/ascii_input_4.data'],
                          # ['tests/data/ascii_input_5.data']
                          ])
def test_read_data(input_data):
    """Test """
    output = read_data(input_data)
    if isinstance(output, list):
        pass
    else:
        raise TypeError
    print(output)


@pytest.mark.parametrize('input_data',
                         [
                             [1., 3., 2., 5.],
                             [1, 3, 2, 5],
                             [1, 10],
                             [10, 1],
                             [0, 0],
                             None,
                             # [1],              # TODO
                             # [1, 2, 3],
                             # [1, 2, 3, 4, 5],
                         ])
def test_list_to_slice(input_data):
    """Test """
    output = list_to_slice(input_data)
    if isinstance(output, slice):
        pass
    elif isinstance(output, tuple) and all(isinstance(item, slice) for item in output):
        pass
    else:
        raise TypeError
    print(output)


@pytest.mark.parametrize('targ_range, out_range, row, col',
                         [
                             ([0, 2, 0, 2], [0, 2, 0, 2], 5, None),
                             ([0, 3, 0, 2], [0, 2, 0, 2], 5, 5),
                             ([0, 2, 0, 2], [0, 2, 0, 3], 5, 5),
                             ([0, 2, 0, 2], [0, 2, 0, 2], 1, 5),
                             ([0, 2, 0, 2], [0, 2, 0, 2], 1, 1),
                             ([0, 2, 0, 2], [0, 2, 0, 2], 0, 0),
                             ([0, 2], [0, 3], 3, 3),
                             ([-1, 2], [0, 2], 3, 3),
                             ([-1, 1], [0, 2], 3, 3),
                             ([0, 2], [0, 2], 1, 1),
                             ([0, 2], [0, 2], 0, 0),
                             ([0, 2], [0, 2], 0, None),
                             ([0, 2], [0, 2], 0, 0),
                             ([0], [0, 2], 4, 4),
                             ([0, 2], [0, 2, 3], 4, 4),
                             ([0, 2, 0, 4], [0, 2, 0, 4], 3, 3),
                             ([0, 2, -2, 2], [0, 2, -2, 2], 3, 3),

                         ])
def test_check_ranges(targ_range, out_range, row, col):
    """Test """
    with pytest.raises(ValueError):
        check_ranges(targ_range, out_range, row, col)

@pytest.mark.skip(reason='!! FIX THIS TEST !!')
@pytest.mark.skipif(not WITH_PYGMO, reason="Package 'pygmo' is not installed.")
@pytest.mark.parametrize('yaml',
                         [
                             'tests/data/calibrate_models.yaml'
                         ])
def test_run_calibration(yaml):
    """Test """
    cfg = io.load(yaml)
    assert isinstance(cfg, dict)

    detector = cfg['ccd_detector']
    assert isinstance(detector, CCD)
    
    pipeline = cfg['pipeline']
    assert isinstance(pipeline, DetectionPipeline)

    processor = Processor(detector, pipeline)
    assert isinstance(processor, Processor)

    simulation = cfg['simulation']
    assert isinstance(simulation, Configuration)

    result = simulation.calibration.run_calibration(processor)
    # assert result == 1         # TODO

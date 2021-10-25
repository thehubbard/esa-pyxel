"""Unittests for the 'Calibration' class."""

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import typing as t
from pathlib import Path

import numpy as np
import pytest

from pyxel.calibration import Calibration
from pyxel.calibration.util import (
    check_ranges,
    list_to_slice,
    read_data,
    read_single_data,
)
from pyxel.configuration import Configuration, load
from pyxel.detectors import CCD
from pyxel.pipelines import DetectionPipeline, Processor

try:
    import pygmo as pg

    WITH_PYGMO = True
except ImportError:
    WITH_PYGMO = False


@pytest.mark.skipif(not WITH_PYGMO, reason="Package 'pygmo' is not installed.")
@pytest.mark.parametrize(
    "yaml",
    [
        "tests/data/calibrate.yaml",
        "tests/data/calibrate_sade.yaml",
        "tests/data/calibrate_sga.yaml",
        "tests/data/calibrate_nlopt.yaml",
        "tests/data/calibrate_models.yaml",
        "tests/data/calibrate_custom_fitness.yaml",
        "tests/data/calibrate_fits.yaml",
    ],
)
def test_set_algo(yaml):
    """Test"""
    cfg = load(yaml)
    calibration = cfg.calibration
    obj = calibration.algorithm.get_algorithm()
    if isinstance(obj, pg.sade):
        pass
    elif isinstance(obj, pg.sga):
        pass
    elif isinstance(obj, pg.nlopt):
        pass
    else:
        raise ReferenceError


@pytest.mark.parametrize(
    "input_data",
    [
        [Path("tests/data/calibrate-data.npy")],
        [Path("tests/data/calibrate-data.txt")],
        [Path("tests/data/ascii_input_3.data")],
        [Path("tests/data/ascii_input_4.data")],
        pytest.param(
            Path("tests/data/calibrate-data.npy"),
            marks=pytest.mark.xfail(raises=TypeError),
        ),
        pytest.param(
            Path("tests/data/calibrate-data.txt"),
            marks=pytest.mark.xfail(raises=TypeError),
        ),
        pytest.param(
            Path("tests/data/ascii_input_1.data"),
            marks=pytest.mark.xfail(raises=TypeError),
        ),
        pytest.param(
            Path("tests/data/ascii_input_2.data"),
            marks=pytest.mark.xfail(raises=TypeError),
        ),
    ],
)
def test_read_data(input_data):
    """Test"""
    output = read_data(input_data)
    assert isinstance(output, list)


@pytest.mark.parametrize(
    "input_data",
    [
        # pytest.param(
        #     [Path("tests/data/calibrate-data.npy")],
        #     marks=pytest.mark.xfail(raises=TypeError),
        # ),
        # pytest.param(
        #     [Path("tests/data/calibrate-data.txt")],
        #     marks=pytest.mark.xfail(raises=TypeError),
        # ),
        # pytest.param(
        #     [Path("tests/data/ascii_input_3.data")],
        #     marks=pytest.mark.xfail(raises=TypeError),
        # ),
        # pytest.param(
        #     [Path("tests/data/ascii_input_4.data")],
        #     marks=pytest.mark.xfail(raises=TypeError),
        # ),
        Path("tests/data/calibrate-data.npy"),
        Path("tests/data/calibrate-data.txt"),
        Path("tests/data/ascii_input_1.data"),
        Path("tests/data/ascii_input_2.data"),
    ],
)
def test_read_single_data(input_data):
    output = read_single_data(input_data)

    assert isinstance(output, np.ndarray)


@pytest.mark.parametrize(
    "input_data",
    [
        [1.0, 3.0, 2.0, 5.0],
        [1, 3, 2, 5],
        [0, 0, 0, 0],
        [2, 4, 1, 3, 2, 5],
        [4.0, 8.0, 1.0, 3.0, 2.0, 5.0],
        None,
        # [1],              # TODO
        # [1, 2, 3],
        # [1, 2, 3, 4, 5],
    ],
)
def test_list_to_slice(input_data):
    """Test"""
    output = list_to_slice(input_data)
    if isinstance(output, slice):
        pass
    elif isinstance(output, tuple) and all(isinstance(item, slice) for item in output):
        pass
    else:
        raise TypeError
    print(output)


@pytest.mark.parametrize(
    "target_range, out_range, rows, cols, readout_times",
    [
        ([], [], 10, 10, None),
        # 2D
        ([0, 5, 0, 10], [0, 5, 0, 10], 5, 10, None),  # No 'readout_times'
        ([0, 5, 0, 10], [10, 15, 10, 20], 5, 10, None),  # No 'readout_times'
        ([0, 5, 0, 10], [0, 5, 0, 10], 5, 10, 0),  # Lowest 'readout_times'
        ([0, 5, 0, 10], [0, 5, 0, 10], 5, 10, 5),  # Highest 'readout_times'
        # 3D
        (
            [0, 5, 0, 10, 0, 20],
            [0, 5, 0, 10, 0, 20],
            10,
            20,
            None,
        ),  # No 'readout_times'
    ],
)
def test_check_range_valid(
    target_range: list,
    out_range: list,
    rows: int,
    cols: int,
    readout_times: t.Optional[int],
):
    """Test valid values for function 'check_range'."""
    check_ranges(
        target_fit_range=target_range,
        out_fit_range=out_range,
        rows=rows,
        cols=cols,
        readout_times=readout_times,
    )


@pytest.mark.parametrize(
    "target_range, out_range, rows, cols, readout_times, exp_error",
    [
        # 'target_range' is neither 2D nor 3D
        pytest.param([0], [0, 5, 0, 10], 5, 10, None, "", id="Target 1 element"),
        pytest.param([0, 5], [0, 5, 0, 10], 5, 10, None, "", id="Target 2 elements"),
        pytest.param([0, 5, 0], [0, 5, 0, 10], 5, 10, None, "", id="Target 3 elements"),
        pytest.param(
            [0, 5, 10, 0, 5], [0, 5, 0, 10], 5, 10, None, "", id="Target 5 elements"
        ),
        # 'out_range' is neither 2D nor 3D
        pytest.param([0, 5, 0, 10], [0], 5, 10, None, "", id="Out 1 element"),
        pytest.param([0, 5, 0, 10], [0, 5], 5, 10, None, "", id="Out 2 element"),
        pytest.param([0, 5, 0, 10], [0, 5, 0], 5, 10, None, "", id="Out 3 element"),
        pytest.param(
            [0, 5, 0, 10], [0, 5, 0, 10, 11], 5, 10, None, "", id="Out 5 element"
        ),
        # Different span for 'target_range' and 'out_range'
        pytest.param(
            [0, 5, 0, 10],
            [0, 6, 0, 10],
            5,
            10,
            None,
            "Fitting ranges have different lengths in 1st dimension",
            id="1D length - too long",
        ),
        pytest.param(
            [0, 5, 0, 10],
            [1, 5, 0, 10],
            5,
            10,
            None,
            "Fitting ranges have different lengths in 1st dimension",
            id="1D length - too short",
        ),
        pytest.param(
            [0, 5, 0, 10],
            [0, 5, 0, 11],
            5,
            10,
            None,
            "Fitting ranges have different lengths in 2nd dimension",
            id="2D length - too long",
        ),
        pytest.param(
            [0, 5, 0, 10],
            [0, 5, 0, 9],
            5,
            10,
            None,
            "Fitting ranges have different lengths in 2nd dimension",
            id="2D length - too short",
        ),
        pytest.param(
            [0, 5, 0, 10, 0, 20],
            [0, 5, 0, 10, 0, 19],
            5,
            10,
            None,
            "Fitting ranges have different lengths in third dimension",
            id="3D length - too short",
        ),
        pytest.param(
            [0, 5, 0, 10, 0, 20],
            [0, 5, 0, 10, 0, 21],
            5,
            10,
            None,
            "Fitting ranges have different lengths in third dimension",
            id="3D length - too long",
        ),
        # ([0, 2], [0, 3], 3, 3, None),
        # ([-1, 2], [0, 2], 3, 3, None),
        # ([-1, 1], [0, 2], 3, 3, None),
        # ([0, 2], [0, 2], 1, 1, None),
        # ([0, 2], [0, 2], 0, 0, None),
        # ([0, 2], [0, 2], 0, 0, None),
        # ([0], [0, 2], 4, 4, None),
        # ([0, 2], [0, 2, 3], 4, 4, None),
        # 2D
        # ([0, 3, 0, 2], [0, 2, 0, 2], 5, 5, None),
        # ([0, 2, 0, 2], [0, 2, 0, 3], 5, 5, None),
        # ([0, 2, 0, 2], [0, 2, 0, 2], 1, 5, None),
        # ([0, 2, 0, 2], [0, 2, 0, 2], 1, 1, None),
        # ([0, 2, 0, 2], [0, 2, 0, 2], 0, 0, None),
        # ([0, 2, 0, 4], [0, 2, 0, 4], 3, 3, None),
        # ([0, 2, -2, 2], [0, 2, -2, 2], 3, 3, None),
    ],
)
def test_check_ranges_invalid(
    target_range: list,
    out_range: list,
    rows: int,
    cols: int,
    readout_times: t.Optional[int],
    exp_error,
):
    """Test"""
    with pytest.raises(ValueError, match=exp_error):
        check_ranges(
            target_fit_range=target_range,
            out_fit_range=out_range,
            rows=rows,
            cols=cols,
            readout_times=readout_times,
        )


@pytest.mark.skip(reason="!! FIX THIS TEST !!")
@pytest.mark.skipif(not WITH_PYGMO, reason="Package 'pygmo' is not installed.")
@pytest.mark.parametrize("yaml", ["tests/data/calibrate_models.yaml"])
def test_run_calibration(yaml):
    """Test"""
    cfg = load(yaml)
    assert isinstance(cfg, Configuration)

    detector = cfg.ccd_detector
    assert isinstance(detector, CCD)

    pipeline = cfg.pipeline
    assert isinstance(pipeline, DetectionPipeline)

    processor = Processor(detector, pipeline)
    assert isinstance(processor, Processor)

    calibration = cfg.calibration
    assert isinstance(calibration, Calibration)

    assert calibration is not None
    result = calibration.run_calibration(processor)
    # assert result == 1         # TODO

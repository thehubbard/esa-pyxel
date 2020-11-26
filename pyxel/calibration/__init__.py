#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""

try:
    import pygmo as _pg

    del _pg
except ImportError:
    raise RuntimeError(
        "Missing package 'pygmo'. Please install it with 'conda install pygmo' or 'pip install pygmo'."
    )

# flake8: noqa
from .util import (
    CalibrationResult,
    CalibrationMode,
    Island,
    AlgorithmType,
    ResultType,
    check_ranges,
    list_to_slice,
    read_data,
)
from .calibration import Calibration, Algorithm, CalibrationMode

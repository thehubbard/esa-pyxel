#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Utility functions."""

import logging
import typing as t

import numpy as np

# flake8: noqa
# from pyxel.util.outputs import image, numpy_array, hist_plot, graph_plot, show_plots
from pyxel.util.jupyxel import (
    display_config,
    display_dict,
    display_model,
    change_modelparam,
    display_array,
    display_detector,
    display_persist,
)

from pyxel.util.memory import get_size, memory_usage_details

__all__ = [
    "convert_to_int",
    "round_convert_to_int",
    "PipelineAborted",
    "Outputs",
    "LogFilter",
    "apply_run_number",
]


class PipelineAborted(Exception):
    """Exception to force the pipeline to stop processing."""

    def __init__(
        self,
        message: t.Optional[str] = None,
        # errors=None
    ):
        super().__init__(message)
        # self.errors = errors


def round_convert_to_int(input_array: np.ndarray) -> np.ndarray:
    """Round list of floats in numpy array and convert to integers.

    Use on data before adding into DataFrame.

    :param input_array: numpy array object OR numpy array (float, int)
    :return:
    """
    array = input_array.astype(float)
    array = np.rint(array)
    array = array.astype(int)
    return array


def convert_to_int(input_array: np.ndarray) -> np.ndarray:
    """Convert numpy array to integer.

    Use on data after getting it from DataFrame.

    :param input_array: numpy array object OR numpy array (float, int)
    :return:
    """
    return input_array.astype(int)


class LogFilter(logging.Filter):
    """TBW."""

    def filter(self, record):
        """Filter or modify record."""
        if record.threadName == "MainThread" or record.threadName.endswith("-0"):
            return True
        else:
            return False

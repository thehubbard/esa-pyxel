"""Utility functions."""
import glob
import numpy as np
from pyxel.util.fitsfile import FitsFile

__all__ = ['update_fits_header', 'apply_run_number', 'convert_to_int', 'round_convert_to_int',
           'FitsFile', 'PipelineAborted']


class PipelineAborted(Exception):
    """Exception to force the pipeline to stop processing."""

    def __init__(self, message: str = None, errors=None):
        """TBW."""
        super().__init__(message)
        self.errors = errors


def round_convert_to_int(input_array: np.ndarray):
    """Round list of floats in numpy array and convert to integers.

    Use on data before adding into DataFrame.

    :param input_array: numpy array object OR numpy array (float, int)
    :return:
    """
    array = input_array.astype(float)
    array = np.rint(array)
    array = array.astype(int)
    return array


def convert_to_int(input_array: np.ndarray):
    """Convert numpy array to integer.

    Use on data after getting it from DataFrame.

    :param input_array: numpy array object OR numpy array (float, int)
    :return:
    """
    return input_array.astype(int)


def update_fits_header(header, key, value):
    """TBW.

    :param header:
    :param key:
    :param value:
    :return:
    """
    if not isinstance(value, (str, int, float)):
        value = '%r' % value
    if isinstance(value, str):
        value = value[0:24]
    if isinstance(key, (list, tuple)):
        key = '/'.join(key)
    key = key.replace('.', '/')[0:36]
    header[key] = value


def apply_run_number(path):
    """Convert the file name numeric placeholder to a unique number.

    :param path:
    :return:
    """
    path_str = str(path)
    if '?' in path_str:
        dir_list = sorted(glob.glob(path_str))
        p_0 = path_str.find('?')
        p_1 = path_str.rfind('?')
        template = path_str[p_0: p_1 + 1]
        path_str = path_str.replace(template, '{:0%dd}' % len(template))
        last_num = 0
        if len(dir_list):
            path_last = dir_list[-1]
            last_num = int(path_last[p_0: p_1 + 1])
        last_num += 1
        path_str = path_str.format(last_num)
    return type(path)(path_str)

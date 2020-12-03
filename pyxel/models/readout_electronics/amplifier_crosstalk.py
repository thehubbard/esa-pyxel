#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
"""Amplifier crosstalk model: https://arxiv.org/abs/1808.00790."""

import typing as t

import numba
import numpy as np
from pyxel.inputs_outputs import load_table
from pathlib import Path
if t.TYPE_CHECKING:
    from pyxel.detectors import Detector


@numba.njit
def flip_array(array: np.ndarray, pos: int) -> np.ndarray:
    """Flip the array for read direction as in starting position 1 or back."""
    if pos == 1:
        return array
    elif pos == 2:
        return np.fliplr(array)
    elif pos == 3:
        return np.flipud(array)
    elif pos == 4:
        return np.fliplr(np.flipud(array))
    else:
        raise ValueError("Unknown readout direction.")


@numba.njit
def get_channel_slices(detector_shape: tuple, channel_matrix: np.array):
    columns = True
    j_x = channel_matrix.shape[0]
    j_y = 1
    if len(channel_matrix.shape) > 1:
        columns = False
        j_y = channel_matrix.shape[1]

    x = detector_shape[0]
    y = detector_shape[1]

    slices = []

    for j in range(1, channel_matrix.size + 1):
        channel_position = np.where(channel_matrix == j)
        if columns:
            channel_slice_x = slice(
                channel_position[0][0] * x // j_x,
                (channel_position[0][0] + 1) * x // j_x,
            )
            channel_slice_y = slice(0, y)
            slices.append([channel_slice_x, channel_slice_y])
        else:
            channel_slice_x = slice(
                channel_position[1][0] * x // j_x,
                (channel_position[1][0] + 1) * x // j_x,
            )
            channel_slice_y = slice(
                channel_position[0][0] * y // j_y,
                (channel_position[0][0] + 1) * y // j_y,
            )
            slices.append([channel_slice_x, channel_slice_y])

    return slices


def get_matrix(coupling_matrix: t.Union[str, Path, list]) -> np.ndarray:
    if isinstance(coupling_matrix, list):
        return np.array(coupling_matrix)
    else:
        return np.array(load_table(coupling_matrix))


@numba.njit(parallel=True)
def crosstalk_signal_ac(
    array: np.ndarray,
    coupling_matrix: np.ndarray,
    channel_matrix: np.ndarray,
    readout_directions: np.ndarray,
):
    amp_number = channel_matrix.size   # number of amplifiers

    slices = get_channel_slices(array.shape, channel_matrix)

    array_copy = array.copy()

    for k in range(amp_number):
        for j in range(amp_number):
            if k != j and coupling_matrix[k][j] != 0:
                s_k = flip_array(array_copy[slices[k][1], slices[k][0]], readout_directions[k])
                s_k_shift = np.hstack((s_k[:, 0:1], s_k[:, 0:-1]))
                delta_s = s_k - s_k_shift
                crosstalk_signal = coupling_matrix[k][j] * flip_array(delta_s, readout_directions[j])
                array[slices[j][1], slices[j][0]] += crosstalk_signal

    return array


@numba.njit(parallel=True)
def crosstalk_signal_dc(
    array: np.ndarray,
    coupling_matrix: np.ndarray,
    channel_matrix: np.ndarray,
    readout_directions: np.ndarray,
):
    amp_number = channel_matrix.size  # number of amplifiers

    slices = get_channel_slices(
        array.shape, channel_matrix
    )  # type: t.List[t.List[slice]]

    array_copy = array.copy()

    for k in range(amp_number):
        for j in range(amp_number):
            if k != j and coupling_matrix[k][j] != 0:
                s_k = flip_array(array_copy[slices[k][1], slices[k][0]], readout_directions[k])
                crosstalk_signal = coupling_matrix[k][j] * flip_array(s_k, readout_directions[j])
                array[slices[j][1], slices[j][0]] += crosstalk_signal

    return array


def dc_crosstalk(
    detector: "Detector", coupling_matrix: t.Union[str, Path, list], channel_matrix: list, readout_directions: list
):
    coupling_matrix = get_matrix(coupling_matrix)  # type: np.ndarray
    channel_matrix = np.array(channel_matrix)  # type: np.ndarray
    readout_directions = np.array(readout_directions)  # type: np.ndarray

    if detector.geometry.row % channel_matrix.shape[0] != 0:
        raise ValueError("Can't split detector array horizontally for a given number of amplifiers.")
    if len(channel_matrix.shape) > 1:
        if detector.geometry.col % channel_matrix.shape[1] != 0:
            raise ValueError("Can't split detector array vertically for a given number of amplifiers.")
    if channel_matrix.size != readout_directions.size:
        raise ValueError("Channel matrix and readout directions arrays not the same size.")

    _ = crosstalk_signal_dc(
        array=detector.signal.array,
        coupling_matrix=coupling_matrix,
        channel_matrix=channel_matrix,
        readout_directions=readout_directions,
    )


def ac_crosstalk(
    detector: "Detector", coupling_matrix: t.Union[str, Path, list], channel_matrix: list, readout_directions: list
):
    coupling_matrix = get_matrix(coupling_matrix)  # type: np.ndarray
    channel_matrix = np.array(channel_matrix)  # type: np.ndarray
    readout_directions = np.array(readout_directions)  # type: np.ndarray

    if detector.geometry.row % channel_matrix.shape[0] != 0:
        raise ValueError("Can't split detector array horizontally for a given number of amplifiers.")
    if len(channel_matrix.shape) > 1:
        if detector.geometry.col % channel_matrix.shape[1] != 0:
            raise ValueError("Can't split detector array vertically for a given number of amplifiers.")
    if channel_matrix.size != readout_directions.size:
        raise ValueError("Channel matrix and readout directions arrays not the same size.")

    _ = crosstalk_signal_ac(
        array=detector.signal.array,
        coupling_matrix=coupling_matrix,
        channel_matrix=channel_matrix,
        readout_directions=readout_directions,
    )

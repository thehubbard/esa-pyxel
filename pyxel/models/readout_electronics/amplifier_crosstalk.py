#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Amplifier crosstalk model: https://arxiv.org/abs/1808.00790."""

import typing as t
from pathlib import Path

import numba
import numpy as np

from pyxel.inputs_outputs import load_table

if t.TYPE_CHECKING:
    from pyxel.detectors import Detector


@numba.njit
def flip_array(array: np.ndarray, direction: int) -> np.ndarray:
    """Flip the array for read direction as in case 1 or back.

    Parameters
    ----------
    array: ndarray
    direction: int

    Returns
    -------
    ndarray
    """
    if direction == 1:
        return array
    elif direction == 2:
        return np.fliplr(array)
    elif direction == 3:
        return np.flipud(array)
    elif direction == 4:
        return np.fliplr(np.flipud(array))
    else:
        raise ValueError("Unknown readout direction.")


@numba.njit
def get_channel_slices(
    shape: tuple, channel_matrix: np.array
) -> t.List[t.List[t.Tuple[t.Any, t.Any]]]:
    """Get pairs of slices that correspond to the given channel matrix in numerical order of channels.

    Parameters
    ----------
    shape: tuple
    channel_matrix: ndarray

    Returns
    -------
    slices
    """
    j_x = channel_matrix.shape[0]
    j_y = 1
    if channel_matrix.ndim == 2:
        j_y = channel_matrix.shape[1]

    delta_x = shape[0] // j_x
    delta_y = shape[1] // j_y

    slices = []

    for j in range(1, channel_matrix.size + 1):
        channel_position = np.argwhere(channel_matrix == j)[0]
        if channel_position.size == 1:
            channel_position = np.append(np.array([0]), channel_position)
        channel_slice_x = (
            channel_position[1] * delta_x,
            (channel_position[1] + 1) * delta_x,
        )
        channel_slice_y = (
            channel_position[0] * delta_y,
            (channel_position[0] + 1) * delta_y,
        )
        slices.append([channel_slice_x, channel_slice_y])

    return slices


def get_matrix(coupling_matrix: t.Union[str, Path, list]) -> np.ndarray:
    """Get the coupling matrix either from configuration input or a file.

    Parameters
    ----------
    coupling_matrix

    Returns
    -------
    ndarray
    """
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
) -> np.ndarray:
    """Apply AC crosstalk signal to an array.

    Parameters
    ----------
    array: ndarray
    coupling_matrix: ndarray
    channel_matrix: ndarray
    readout_directions: ndarray

    Returns
    -------
    array: ndarray
    """
    amp_number = channel_matrix.size  # number of amplifiers

    slices = get_channel_slices(
        shape=array.shape, channel_matrix=channel_matrix
    )  # type: t.List[t.List[t.Tuple[t.Any, t.Any]]]

    array_copy = array.copy()

    for k in range(amp_number):
        for j in range(amp_number):
            if k != j and coupling_matrix[k][j] != 0:
                s_k = flip_array(
                    array_copy[
                        slices[k][1][0] : slices[k][1][1],
                        slices[k][0][0] : slices[k][0][1],
                    ],
                    readout_directions[k],
                )
                s_k_shift = np.hstack((s_k[:, 0:1], s_k[:, 0:-1]))
                delta_s = s_k - s_k_shift
                crosstalk_signal = coupling_matrix[k][j] * flip_array(
                    delta_s, readout_directions[j]
                )
                array[
                    slices[j][1][0] : slices[j][1][1], slices[j][0][0] : slices[j][0][1]
                ] += crosstalk_signal

    return array


@numba.njit(parallel=True)
def crosstalk_signal_dc(
    array: np.ndarray,
    coupling_matrix: np.ndarray,
    channel_matrix: np.ndarray,
    readout_directions: np.ndarray,
) -> np.ndarray:
    """Apply DC crosstalk signal to an array.

    Parameters
    ----------
    array: ndarray
    coupling_matrix: ndarray
    channel_matrix: ndarray
    readout_directions: ndarray

    Returns
    -------
    array: ndarray
    """
    amp_number = channel_matrix.size  # number of amplifiers

    slices = get_channel_slices(
        shape=array.shape, channel_matrix=channel_matrix
    )  # type: t.List[t.List[t.Tuple[t.Any, t.Any]]]

    array_copy = array.copy()

    for k in range(amp_number):
        for j in range(amp_number):
            if k != j and coupling_matrix[k][j] != 0:
                s_k = flip_array(
                    array_copy[
                        slices[k][1][0] : slices[k][1][1],
                        slices[k][0][0] : slices[k][0][1],
                    ],
                    readout_directions[k],
                )
                crosstalk_signal = coupling_matrix[k][j] * flip_array(
                    s_k, readout_directions[j]
                )
                array[
                    slices[j][1][0] : slices[j][1][1], slices[j][0][0] : slices[j][0][1]
                ] += crosstalk_signal

    return array


def dc_crosstalk(
    detector: "Detector",
    coupling_matrix: t.Union[str, Path, list],
    channel_matrix: list,
    readout_directions: list,
) -> None:
    """Apply DC crosstalk signal to detector signal.

    Parameters
    ----------
    detector: Detector
    coupling_matrix: ndarray
    channel_matrix: ndarray
    readout_directions: ndarray

    Returns
    -------
    None
    """
    cpl_matrix = get_matrix(coupling_matrix)  # type: np.ndarray
    ch_matrix = np.array(channel_matrix)  # type: np.ndarray
    directions = np.array(readout_directions)  # type: np.ndarray

    if detector.geometry.row % ch_matrix.shape[0] != 0:
        raise ValueError(
            "Can't split detector array horizontally for a given number of amplifiers."
        )
    if len(ch_matrix.shape) > 1:
        if detector.geometry.col % ch_matrix.shape[1] != 0:
            raise ValueError(
                "Can't split detector array vertically for a given number of amplifiers."
            )
    if ch_matrix.size != directions.size:
        raise ValueError(
            "Channel matrix and readout directions arrays not the same size."
        )

    _ = crosstalk_signal_dc(
        array=detector.signal.array,
        coupling_matrix=cpl_matrix,
        channel_matrix=ch_matrix,
        readout_directions=directions,
    )


def ac_crosstalk(
    detector: "Detector",
    coupling_matrix: t.Union[str, Path, list],
    channel_matrix: list,
    readout_directions: list,
) -> None:
    """Apply AC crosstalk signal to detector signal.

    Parameters
    ----------
    detector: Detector
    coupling_matrix: ndarray
    channel_matrix: ndarray
    readout_directions: ndarray

    Returns
    -------
    None
    """
    cpl_matrix = get_matrix(coupling_matrix)  # type: np.ndarray
    ch_matrix = np.array(channel_matrix)  # type: np.ndarray
    directions = np.array(readout_directions)  # type: np.ndarray

    if detector.geometry.row % ch_matrix.shape[0] != 0:
        raise ValueError(
            "Can't split detector array horizontally for a given number of amplifiers."
        )
    if len(ch_matrix.shape) > 1:
        if detector.geometry.col % ch_matrix.shape[1] != 0:
            raise ValueError(
                "Can't split detector array vertically for a given number of amplifiers."
            )
    if ch_matrix.size != directions.size:
        raise ValueError(
            "Channel matrix and readout directions arrays not the same size."
        )

    _ = crosstalk_signal_ac(
        array=detector.signal.array,
        coupling_matrix=cpl_matrix,
        channel_matrix=ch_matrix,
        readout_directions=directions,
    )

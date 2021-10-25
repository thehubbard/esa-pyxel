#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Phase-pulse processing model."""

import numba
import numpy as np

from pyxel.detectors import MKID


def pulse_processing(
    detector: MKID,
    wavelength: float,
    responsivity: float,
) -> None:
    """TBW."""
    geo = detector.geometry
    ch = detector.characteristics
    ph = detector.photon

    detector_charge = np.zeros(
        (geo.row, geo.col)
    )  # all pixels has zero charge by default
    photon_rows, photon_cols = ph.array.shape
    detector_charge[slice(0, photon_rows), slice(0, photon_cols)] = (
        ph.array * ch.qe * ch.eta
    )
    charge_number = detector_charge.flatten()  # the average charge numbers per pixel
    where_non_zero = np.where(charge_number > 0.0)
    charge_number = charge_number[where_non_zero]
    size = charge_number.size

    init_ver_pix_position = geo.vertical_pixel_center_pos_list()[where_non_zero]
    init_hor_pix_position = geo.horizontal_pixel_center_pos_list()[where_non_zero]

    detector.charge.add_charge(
        particle_type="e",
        particles_per_cluster=charge_number,
        init_energy=np.zeros(size),
        init_ver_position=init_ver_pix_position,
        init_hor_position=init_hor_pix_position,
        init_z_position=np.zeros(size),
        init_ver_velocity=np.zeros(size),
        init_hor_velocity=np.zeros(size),
        init_z_velocity=np.zeros(size),
    )

    array = np.zeros((geo.row, geo.col))

    charge_per_pixel = detector.charge.get_values(quantity="number")
    charge_pos_ver = detector.charge.get_values(quantity="position_ver")
    charge_pos_hor = detector.charge.get_values(quantity="position_hor")

    pixel_index_ver = np.floor_divide(charge_pos_ver, geo.pixel_vert_size).astype(int)
    pixel_index_hor = np.floor_divide(charge_pos_hor, geo.pixel_horz_size).astype(int)

    # Changing = to += since charge dataframe is reset, the pixel array need to be
    # incremented, we can't do the whole operation on each iteration
    detector.phase.array += df_to_array(
        array=array,
        charge_per_pixel=charge_per_pixel,
        pixel_index_ver=pixel_index_ver,
        pixel_index_hor=pixel_index_hor,
    ).astype(np.int32)

    detector.phase.array = detector.phase.array * wavelength * 2.5e2 / responsivity
    detector.phase.array = detector.phase.array.astype("float64")


@numba.jit(nopython=True)
def df_to_array(
    array: np.ndarray,
    charge_per_pixel: np.ndarray,
    pixel_index_ver: np.ndarray,
    pixel_index_hor: np.ndarray,
) -> np.ndarray:
    """TBW."""

    for i, charge_value in enumerate(charge_per_pixel):
        array[pixel_index_ver[i], pixel_index_hor[i]] += charge_value
    return array

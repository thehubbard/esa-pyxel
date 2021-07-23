#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import typing as t

import numba
import numpy as np

import pyxel.models.charge_transfer.arctic_cti.arctic_numba as ac_numba
from pyxel.detectors import CCD


def arctic_add_numba(
    detector: CCD,
    well_fill_power: float,
    # instant_traps: t.Sequence[t.Dict[str, float]],
    trap_1_density: float,
    trap_1_release_time_scale: float,
    express: int = 0,
) -> None:
    image_2d = np.asarray(detector.pixel.array, dtype=float)  # type: np.ndarray

    ccd = ac_numba.CCD(
        n_phases=1,
        fraction_of_traps_per_phase=np.array([1.0], dtype=np.float64),
        full_well_depth=np.array([detector.characteristics.fwc], dtype=np.float64),
        well_notch_depth=np.array([0.0], dtype=np.float64),
        well_fill_power=np.array([well_fill_power], dtype=np.float64),
        well_bloom_level=np.array([detector.characteristics.fwc], dtype=np.float64),
    )
    parallel_roe = ac_numba.ROE(dwell_times=np.array([1.0], dtype=np.float64))

    instant_traps = [
        {"density": trap_1_density, "release_timescale": trap_1_release_time_scale}
    ]  # type: t.Sequence[t.Mapping[str, float]]

    # Build the traps
    n_traps = len(instant_traps)

    densities = []  # type: t.List[float]
    release_timescales = []  # type: t.List[float]

    for trap_info in instant_traps:
        density = trap_info["density"]  # type: float
        release_timescale = trap_info["release_timescale"]  # type: float

        densities.append(density)
        release_timescales.append(release_timescale)

    traps = ac_numba.TrapsInstantCapture(
        density_1d=np.array(densities, dtype=np.float64),
        release_timescale_1d=np.array(release_timescales, dtype=np.float64),
        surface_1d=np.array([False] * n_traps, dtype=np.bool_),
    )

    traps_lst = numba.typed.List()
    traps_lst.append(traps)

    image_cti_added = ac_numba.add_cti(
        image_2d=image_2d,
        parallel_traps=traps_lst,
        parallel_ccd=ccd,
        parallel_roe=parallel_roe,
        parallel_express=express,
        # serial_roe=serial_roe,
    )

    detector.pixel.array = image_cti_added


def arctic_remove_numba(
    detector: CCD,
    well_fill_power: float,
    instant_traps: t.Sequence[t.Dict[str, float]],
    num_iterations: int,
    express: int = 0,
) -> None:
    image_2d = np.asarray(detector.pixel.array, dtype=float)  # type: np.ndarray

    ccd = ac_numba.CCD(
        n_phases=1,
        fraction_of_traps_per_phase=np.array([1.0], dtype=np.float64),
        full_well_depth=np.array([detector.characteristics.fwc], dtype=np.float64),
        well_notch_depth=np.array([0.0], dtype=np.float64),
        well_fill_power=np.array([well_fill_power], dtype=np.float64),
        well_bloom_level=np.array([detector.characteristics.fwc], dtype=np.float64),
    )
    parallel_roe = ac_numba.ROE(dwell_times=np.array([1.0], dtype=np.float64))

    # Build the traps
    n_traps = len(instant_traps)

    densities = []  # type: t.List[float]
    release_timescales = []  # type: t.List[float]

    for trap_info in instant_traps:
        density = trap_info["density"]  # type: float
        release_timescale = trap_info["release_timescale"]  # type: float

        densities.append(density)
        release_timescales.append(release_timescale)

    traps = ac_numba.TrapsInstantCapture(
        density_1d=np.array(densities, dtype=np.float64),
        release_timescale_1d=np.array(release_timescales, dtype=np.float64),
        surface_1d=np.array([False] * n_traps, dtype=np.bool_),
    )

    traps_lst = numba.typed.List()
    traps_lst.append(traps)

    image_cti_removed = ac_numba.remove_cti(
        image_2d=image_2d,
        iterations=num_iterations,
        parallel_traps=traps_lst,
        parallel_ccd=ccd,
        parallel_roe=parallel_roe,
        parallel_express=express,
        # serial_roe=serial_roe,
    )

    detector.pixel.array = image_cti_removed

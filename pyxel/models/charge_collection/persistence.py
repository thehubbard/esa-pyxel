#  Copyright (c) YOUR COPYRIGHT HERE

"""
persistence module for the PyXel simulation.

This module is used in charge_collection

-------------------------------------------------------------------------------

+--------------+----------------------------------+---------------------------+
| Author       | Name                             | Creation                  |
+--------------+----------------------------------+---------------------------+
| You          | charge_collection                | 05/20/2021                |
+--------------+----------------------------------+---------------------------+

+-----------------+-------------------------------------+---------------------+
| Contributor     | Name                                | Creation            |
+-----------------+-------------------------------------+---------------------+
| Name            | filename                            | -                   |
+-----------------+-------------------------------------+---------------------+

This module can be found in pyxel/models/charge_collection/persistence.py

Algorithm
=========

The persistence model take as input the total number of detrapped charges
(persistence map with a 10000 seconds soak at more than the FWC) and a map
which is the previous one divided by full well capacity. This gives the total
amount of trap per charge.

At each iteration of the pipeline, the model will first check if there is a
'persistence' entry in the memore of the detector and if not will create it
(happens only at the first iteration).
Then, it will compute the amount of trapped charges in this iteration, add it
to the memory of the detector and then will remove this amount from the pixel array.

Default parameter
=================

The defaults parameter are the one derived from the characterization of the ENG grade
detector for MOONS (H4RG).

+----------------+-------------+
| Time constants | Proportions |
+================+=============+
| 1              | 0.307       |
+----------------+-------------+
| 10             | 0.175       |
+----------------+-------------+
| 100            | 0.188       |
+----------------+-------------+
| 1000           | 0.136       |
+----------------+-------------+
| 10000          | 0.194       |
+----------------+-------------+

Code example
============

.. code-block:: python

    # To access the memory of the detector with the persistence map

    detector._memory["persistence"]

    # This dictionnary consist of 5 entries (one per time constant)

.. literalinclude:: pyxel/models/charge_collection/persistence.py
    :language: python
    :linenos:
    :lines: 84-87

Model reference in the YAML config file
=======================================

.. code-block:: yaml

    pipeline:

      # Persistence model based on MOONS detector (H4RG) measurements
      - name: persistence
        func: pyxel.models.charge_collection.persistence.current_persistence
        enabled: true
        arguments:
          trap_timeconstants: [1, 10, 100, 1000, 10000]
          trap_densities: data/fits/20210408121614_20210128_ENG20370_AUTOCHAR-Persistence_FitTrapDensityMap.fits
          trap_max: data/fits/20210408093114_20210128_ENG20370_AUTOCHAR-Persistence_FitMaximumTrapMap.fits
          trap_proportions: [0.307, 0.175, 0.188, 0.136, 0.194]

Useful links
============

Persistence paper from S. Tulloch
https://arxiv.org/abs/1908.06469

ReadTheDocs documentation
https://sphinx-rtd-theme.readthedocs.io/en/latest/index.html

.. todo::

   - Add temperature dependency
   - Add default -flat?- trap maps

"""

import logging
import typing as t
from dataclasses import dataclass
from pathlib import Path

import numba
import numpy as np
from astropy.io import fits

from pyxel.data_structure import Persistence
from pyxel.detectors import CMOS
from pyxel.util import load_cropped_and_aligned_image
from typing_extensions import Literal
from numba.extending import overload
from numba import types
from numba.core import errors
from numba.np.numpy_support import type_can_asarray, is_nonelike


@dataclass
class SimpleTrap:
    """Define a simple ``Trap``.

    Parameters
    ----------
    density : float
        Unit: N/A
    time_constant : float
        Unit: s
    """

    density: float
    time_constant: float


def get_trapped_charge_name(param1: float, param2: float) -> str:
    """Create a name for the trapped charges based on 'param1' and 'param2'.

    Parameters
    ----------
    param1 : float
    param2 : float

    Returns
    -------
    str
        New name for the trapped charges.
    """
    return f"trappedCharges_{param1}-{param2}"


def create_persistence(
    traps: t.Sequence[SimpleTrap], num_rows: int, num_cols: int
) -> t.Mapping[str, np.ndarray]:
    """Create new empty trapped charges.

    Parameters
    ----------
    traps : sequence of Trap
    num_rows : int
    num_cols : int

    Returns
    -------
    Mapping
        New trapped charges.
    """
    persistence = {}  # type: t.Dict[str, np.ndarray]

    for trap in traps:  # type: SimpleTrap
        entry = get_trapped_charge_name(param1=trap.density, param2=trap.time_constant)
        persistence[entry] = np.zeros((num_rows, num_cols))

    return persistence


def add_persistence(
    traps: t.Sequence[SimpleTrap],
    data_2d: np.ndarray,
    persistence: t.Mapping[str, np.ndarray],
) -> t.Tuple[np.ndarray, t.Mapping[str, np.ndarray]]:
    """Add simple persistence.

    Parameters
    ----------
    traps : sequence of Trap
    data_2d : ndarray
        A 2d array. Unit: electron
    persistence : Mapping
        Trapped charges to add.

    Returns
    -------
    ndarray, Mapping
        A tuple with a new 2D array and new trapped charges.
    """
    new_persistence = {}  # t.Mapping[str, np.ndarray]
    new_data_2d = data_2d  # type: np.ndarray

    for trap in traps:  # type: SimpleTrap
        previous_data_2d = new_data_2d  # type: np.ndarray

        # Get trapped charges
        entry = get_trapped_charge_name(param1=trap.density, param2=trap.time_constant)
        trapped_charges = persistence[entry]  # type: np.ndarray

        # Trap density is a scalar for now, in the future we could feed maps?
        # the delta t is fixed to 0.5 s, need to find a way to avoid problem of divergence
        new_trapped_charges = trapped_charges + (0.5 / trap.time_constant) * (
            previous_data_2d * trap.density - trapped_charges
        )

        # Remove the trapped charges from the pixel
        new_data_2d = previous_data_2d - new_trapped_charges

        # Use new trapped charges
        new_persistence[entry] = new_trapped_charges

    return new_data_2d, new_persistence


def simple_persistence(
    detector: CMOS,
    trap_timeconstants: t.Sequence[float],
    trap_densities: t.Sequence[float],
) -> None:
    """Trapping/detrapping charges."""
    # Validation
    if not len(trap_timeconstants) == len(trap_densities):
        raise ValueError(
            "Expecting same number of elements for parameters 'trap_timeconstants' "
            "and 'trap_densities'"
        )

    if len(trap_timeconstants) == 0:
        raise ValueError(
            "Expecting at least one 'trap_timeconstants' and 'trap_densities'"
        )

    # Conversion
    traps = [
        SimpleTrap(density=density, time_constant=time_constant)
        for density, time_constant in zip(trap_densities, trap_timeconstants)
    ]  # type: t.Sequence[SimpleTrap]

    geometry = detector.geometry

    if "persistence" not in detector._memory:
        persistence = create_persistence(
            traps=traps,
            num_rows=geometry.row,
            num_cols=geometry.col,
        )  # type: t.Mapping[str, np.ndarray]

        detector._memory["persistence"] = persistence

    else:
        new_pixel_2d, new_persistence = add_persistence(
            traps=traps,
            data_2d=detector.pixel.array,
            persistence=detector._memory["persistence"],
        )

        detector.pixel.array = new_pixel_2d

        # Replace old trapped charges map in the detector's memory
        detector._memory["persistence"].update(new_persistence)


def current_persistence(
    detector: CMOS,
    trap_timeconstants: list,
    trap_densities: str,
    trap_max: str,
    trap_proportions: list,
) -> None:
    """Trapping/detrapping charges."""
    logging.info("Persistence")
    # If the file for trap density is correct open it and use it
    # otherwise I need to define a default trap density map

    # Extract trap density / full well
    trap_densities_2d = fits.open(trap_densities)[0].data[
        : detector.geometry.row, : detector.geometry.col
    ]  # type: np.ndarray
    trap_densities_2d[np.where(trap_densities_2d < 0)] = 0

    # Extract the max amount of trap by long soak
    trap_max_2d = fits.open(trap_max)[0].data[
        : detector.geometry.row, : detector.geometry.col
    ]
    trap_max_2d[np.where(trap_max_2d < 0)] = 0

    # If there is no entry for persistence in the memory of the detector
    # create one
    if "persistence" not in detector._memory.keys():
        detector._memory["persistence"] = dict()
        for trap_proportion, trap_timeconstant in zip(
            trap_proportions, trap_timeconstants
        ):
            entry = "".join(
                ["trappedCharges_", str(trap_proportion), "-", str(trap_timeconstant)]
            )
            detector._memory["persistence"].update(
                {entry: np.zeros((detector.geometry.row, detector.geometry.col))}
            )
            trapped_charges = detector._memory["persistence"][entry]

    # For each trap population
    for trap_proportion, trap_timeconstant in zip(trap_proportions, trap_timeconstants):
        # Get the correct persistence traps entry
        entry = "".join(
            ["trappedCharges_", str(trap_proportion), "-", str(trap_timeconstant)]
        )

        # Select the trapped charges array
        trapped_charges = detector._memory["persistence"][entry]

        # Time for reading a frame
        # delta_t = (detector.geometry.row * detector.geometry.col)/detector.characteristics.readout_freq
        delta_t = detector.time_step

        # Computer trapped charge for this increament of time
        # Time factor is the integration time divided by the time constant (1, 10, 100, 1000, 10000)
        time_factor = delta_t / trap_timeconstant

        # Amount of charges trapped per unit of full well
        max_charges = trap_densities_2d * trap_proportion

        # Maximum of amount of charges trapped
        fw_trap = trap_max_2d * trap_proportion

        diff = time_factor * (
            max_charges * detector.pixel.array * np.exp(-time_factor) - trapped_charges
        )
        # Compute trapped charges
        trapped_charges = trapped_charges + time_factor * (
            max_charges * detector.pixel.array * np.exp(-time_factor) - trapped_charges
        )

        # When the amount of trapped charges is superior to the maximum of available traps, set to max
        trapped_charges[np.where(trapped_charges > fw_trap)] = max_charges[
            np.where(trapped_charges > fw_trap)
        ]
        # Can't have a negative amount of charges trapped
        trapped_charges[np.where(trapped_charges < 0)] = 0

        # Remove the trapped charges from the pixel
        # detector.pixel.array -= trapped_charges.astype(np.int32)
        detector.pixel.array -= diff

        # Replace old trapped charges map in the detector's memory
        detector._memory["persistence"][entry] = trapped_charges

    return None


def optimized_persistence(
    detector: CMOS,
    trap_time_constants: t.Sequence[float],
    trap_proportions: t.Sequence[float],
    trap_densities_filename: t.Union[Path, str],
    trap_max_filename: t.Union[Path, str],
    trap_densities_position: t.Tuple[int, int] = (0, 0),
    trap_densities_align: t.Optional[
        Literal["center", "top_left", "top_right", "bottom_left", "bottom_right"]
    ] = None,
    trap_max_position: t.Tuple[int, int] = (0, 0),
    trap_max_align: t.Optional[
        Literal["center", "top_left", "top_right", "bottom_left", "bottom_right"]
    ] = None,
) -> None:
    """

    Parameters
    ----------
    detector
    trap_time_constants
    trap_proportions
    trap_densities_filename
    trap_max_filename
    trap_densities_position
    trap_densities_align
    trap_max_position
    trap_max_align

    Returns
    -------

    """
    # If the file for trap density is correct open it and use it
    # otherwise I need to define a default trap density map
    shape = detector.pixel.shape
    densities_position_y, densities_position_x = trap_densities_position
    max_position_y, max_position_x = trap_max_position

    # Extract trap density / full well
    trap_densities_2d = load_cropped_and_aligned_image(
        shape=shape,
        filename=trap_densities_filename,
        position_x=densities_position_x,
        position_y=densities_position_y,
        align=trap_densities_align,
        allow_smaller_array=False,
    )
    # Extract the max amount of trap by long soak
    trap_max_2d = load_cropped_and_aligned_image(
        shape=shape,
        filename=trap_max_filename,
        position_x=max_position_x,
        position_y=max_position_y,
        align=trap_max_align,
        allow_smaller_array=False,
    )

    trap_densities_2d[np.where(trap_densities_2d < 0)] = 0
    trap_max_2d[np.where(trap_max_2d < 0)] = 0

    if not detector.has_persistence():
        detector.persistence = Persistence(
            trap_time_constants=trap_time_constants,
            trap_proportions=trap_proportions,
            geometry=detector.pixel.shape,
        )

    new_pixel_array, new_all_trapped_charge = compute_persistence(
        pixel_array=detector.pixel.array,
        all_trapped_charge=detector.persistence.trapped_charge_array,
        trap_proportions=np.array(trap_proportions),
        trap_time_constants=np.array(trap_time_constants),
        trap_densities_2d=trap_densities_2d,
        trap_max_2d=trap_max_2d,
        delta_t=detector.time_step,
    )

    detector.pixel.array = new_pixel_array
    detector.persistence.trapped_charge_array = new_all_trapped_charge


@numba.njit(fastmath=True)
def compute_persistence(
    pixel_array: np.ndarray,
    all_trapped_charge: np.ndarray,
    trap_proportions: np.ndarray,
    trap_time_constants: np.ndarray,
    trap_densities_2d: np.ndarray,
    trap_max_2d: np.ndarray,
    delta_t: float,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    pixel_array
    all_trapped_charge
    trap_proportions
    trap_time_constants
    trap_densities_2d
    trap_max_2d
    delta_t

    Returns
    -------

    """
    for i, trapped_charge in enumerate(all_trapped_charge):

        # Computer trapped charge for this increment of time
        # Time factor is the integration time divided by the time constant (1, 10, 100, 1000, 10000)
        time_factor = delta_t / trap_time_constants[i]

        # Amount of charges trapped per unit of full well
        densities = trap_densities_2d * trap_proportions[i]

        # Maximum of amount of charges trapped
        fw_trap = trap_max_2d * trap_proportions[i]

        # Compute trapped charges
        diff = time_factor * (
            densities * pixel_array * np.exp(-time_factor) - trapped_charge
        )
        trapped_charge += diff

        # When the amount of trapped charges is superior to the maximum of available traps, set to max
        # Can't have a negative amount of charges trapped
        trapped_charge = clip_between_zero_and_max(array=trapped_charge, max_2d=fw_trap)

        # Remove the trapped charges from the pixel
        # detector.pixel.array -= trapped_charges.astype(np.int32)
        pixel_array -= diff

        all_trapped_charge[i] = trapped_charge

    return pixel_array, all_trapped_charge


@numba.njit(fastmath=True)
def clip_between_zero_and_max(array: np.ndarray, max_2d: np.ndarray) -> np.ndarray:
    """Clip input array between 0 and array of maximum values.

    Parameters
    ----------
    array: ndarray
    max_2d: ndarray

    Returns
    -------
    ndarray
    """
    n, m = array.shape
    for i in range(n):
        for j in range(m):
            if array[i, j] > max_2d[i, j]:
                array[i, j] = max_2d[i, j]
            if array[i, j] < 0:
                array[i, j] = 0
    return array

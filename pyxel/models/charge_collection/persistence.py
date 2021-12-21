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
        func: pyxel.models.charge_collection.persistence.persistence
        enabled: true
        arguments:
          trap_time_constants: [1, 10, 100, 1000, 10000]
          trap_densities_filename: data/fits/20210408121614_20210128_ENG20370_AUTOCHAR-Persistence_FitTrapDensityMap.fits
          trap_capacities_filename: data/fits/20210408093114_20210128_ENG20370_AUTOCHAR-Persistence_FitMaximumTrapMap.fits
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

import typing as t
from pathlib import Path

import numba
import numpy as np
from typing_extensions import Literal

from pyxel.data_structure import Persistence, SimplePersistence
from pyxel.detectors import CMOS
from pyxel.util import load_cropped_and_aligned_image


def simple_persistence(
    detector: CMOS,
    trap_time_constants: t.Sequence[float],
    trap_densities: t.Sequence[float],
    trap_capacities: t.Sequence[float],
) -> None:

    if not len(trap_time_constants) == len(trap_densities):
        raise ValueError(
            "Expecting same number of elements for parameters 'trap_timeconstants' "
            "and 'trap_densities'"
        )

    if len(trap_time_constants) == 0:
        raise ValueError(
            "Expecting at least one 'trap_timeconstants' and 'trap_densities'"
        )

    if not detector.has_persistence():
        detector.persistence = SimplePersistence(
            trap_time_constants=trap_time_constants,
            trap_densities=trap_densities,
            trap_capacities=trap_capacities,
            geometry=detector.pixel.shape,
        )

    new_pixel_array, new_all_trapped_charge = compute_simple_persistence(
        pixel_array=detector.pixel.array,
        all_trapped_charge=detector.persistence.trapped_charge_array,
        trap_densities=np.array(trap_densities),
        trap_time_constants=np.array(trap_time_constants),
        trap_capacities=np.array(trap_capacities),
        delta_t=detector.time_step,
    )

    detector.pixel.array = new_pixel_array
    detector.persistence.trapped_charge_array = new_all_trapped_charge


@numba.njit(fastmath=True)
def compute_simple_persistence(
        pixel_array: np.ndarray,
        all_trapped_charge: np.ndarray,
        trap_densities: np.ndarray,
        trap_time_constants: np.ndarray,
        trap_capacities: np.ndarray,
        delta_t: float,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    pixel_array
    all_trapped_charge
    trap_densities
    trap_time_constants
    trap_capacities
    delta_t

    Returns
    -------

    """
    for i, trapped_charge in enumerate(all_trapped_charge):
        # Computer trapped charge for this increment of time
        # Time factor is the integration time divided by the time constant (1, 10, 100, 1000, 10000)
        time_factor = delta_t / trap_time_constants[i]

        # Compute trapped charges
        diff = time_factor * (
                trap_densities[i] * pixel_array - trapped_charge
        )

        diff = clip_between_zero_and_max(array=diff, max=trap_capacities[i])

        trapped_charge += diff

        # When the amount of trapped charges is superior to the maximum of available traps, set to max
        # Can't have a negative amount of charges trapped
        trapped_charge = clip_between_zero_and_max_2d(array=trapped_charge, max=trap_capacities[i])

        # Remove the trapped charges from the pixel
        # detector.pixel.array -= trapped_charges.astype(np.int32)
        pixel_array -= diff

        all_trapped_charge[i] = trapped_charge

    return pixel_array, all_trapped_charge


def persistence(
    detector: CMOS,
    trap_time_constants: t.Sequence[float],
    trap_proportions: t.Sequence[float],
    trap_densities_filename: t.Union[Path, str],
    trap_capacities_filename: t.Union[Path, str],
    trap_densities_position: t.Tuple[int, int] = (0, 0),
    trap_densities_align: t.Optional[
        Literal["center", "top_left", "top_right", "bottom_left", "bottom_right"]
    ] = None,
    trap_capacities_position: t.Tuple[int, int] = (0, 0),
    trap_capacities_align: t.Optional[
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
    trap_capacities_filename
    trap_densities_position
    trap_densities_align
    trap_capacities_position
    trap_capacities_align

    Returns
    -------

    """
    # If the file for trap density is correct open it and use it
    # otherwise I need to define a default trap density map
    shape = detector.pixel.shape
    densities_position_y, densities_position_x = trap_densities_position
    capacities_position_y, capacities_position_x = trap_capacities_position

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
    trap_capacities_2d = load_cropped_and_aligned_image(
        shape=shape,
        filename=trap_capacities_filename,
        position_x=capacities_position_x,
        position_y=capacities_position_y,
        align=trap_capacities_align,
        allow_smaller_array=False,
    )

    trap_densities_2d[np.where(trap_densities_2d < 0)] = 0
    trap_capacities_2d[np.where(trap_capacities_2d < 0)] = 0

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
        trap_capacities_2d=trap_capacities_2d,
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
    trap_capacities_2d: np.ndarray,
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
    trap_capacities_2d
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
        fw_trap = trap_capacities_2d * trap_proportions[i]

        # Compute trapped charges
        diff = time_factor * (
            densities * pixel_array * np.exp(-time_factor) - trapped_charge
        )

        diff = clip_between_zero_and_max_2d(array=diff, max_2d=fw_trap - trapped_charge)

        trapped_charge += diff

        # When the amount of trapped charges is superior to the maximum of available traps, set to max
        # Can't have a negative amount of charges trapped
        trapped_charge = clip_between_zero_and_max_2d(array=trapped_charge, max_2d=fw_trap)

        # Remove the trapped charges from the pixel
        # detector.pixel.array -= trapped_charges.astype(np.int32)
        pixel_array -= diff

        all_trapped_charge[i] = trapped_charge

    return pixel_array, all_trapped_charge


@numba.njit(fastmath=True)
def clip_between_zero_and_max_2d(array: np.ndarray, max_2d: np.ndarray) -> np.ndarray:
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


@numba.njit(fastmath=True)
def clip_between_zero_and_max(array: np.ndarray, max: float) -> np.ndarray:
    """Clip input array between 0 and array of maximum values.

    Parameters
    ----------
    array: ndarray
    max: float

    Returns
    -------
    ndarray
    """
    n, m = array.shape
    for i in range(n):
        for j in range(m):
            if array[i, j] > max:
                array[i, j] = max
            if array[i, j] < 0:
                array[i, j] = 0
    return array

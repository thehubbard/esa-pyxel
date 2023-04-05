#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
"""Persistence classes."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class Trap:
    """Trap dataclass."""

    time_constant: float
    proportion: float
    charge: np.ndarray


class Persistence:
    """Persistence class."""

    def __init__(
        self,
        trap_time_constants: Sequence[float],
        trap_proportions: Sequence[float],
        geometry: tuple[int, int],
    ):
        traps: Sequence[Trap] = [
            Trap(
                time_constant=time_constant,
                proportion=trap_proportion,
                charge=np.zeros(geometry),
            )
            for time_constant, trap_proportion in sorted(
                zip(trap_time_constants, trap_proportions)
            )
        ]
        self._trap_list = traps
        self._trapped_charge_array = np.zeros((len(traps), geometry[0], geometry[1]))
        self._trap_time_constants = np.array(trap_time_constants)
        self._trap_proportions = np.array(trap_proportions)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"{cls_name}<number_of_traps={len(self._trap_list)}>"

    @property
    def trap_list(self) -> Sequence[Trap]:
        """List of traps."""
        return self._trap_list

    @property
    def trapped_charge_array(self) -> np.ndarray:
        """Return an array of trapped charge."""
        # for i, trap in enumerate(self.trap_list):
        #     out[i] = trap.charge
        return self._trapped_charge_array

    @trapped_charge_array.setter
    def trapped_charge_array(self, charge_3d: np.ndarray) -> None:
        """Set an array of trapped charge."""
        for i, trap in enumerate(self.trap_list):
            if not charge_3d[i].shape == trap.charge.shape:
                raise ValueError(
                    "Mismatch in shapes between saved trapped charge and input charge."
                )
            trap.charge = charge_3d[i]
        self._trapped_charge_array = charge_3d

    @property
    def trap_time_constants(self) -> np.ndarray:
        """Return trap time constants."""
        return self._trap_time_constants

    @property
    def trap_proportions(self) -> np.ndarray:
        """Return trap proportions."""
        return self._trap_proportions


@dataclass
class SimpleTrap:
    """Simple trap dataclass."""

    time_constant: float
    density: float
    charge: np.ndarray


class SimplePersistence:
    """Simple persistence dataclass."""

    def __init__(
        self,
        trap_time_constants: Sequence[float],
        trap_densities: Sequence[float],
        geometry: tuple[int, int],
    ):
        traps: Sequence[SimpleTrap] = [
            SimpleTrap(
                time_constant=time_constant,
                density=trap_density,
                charge=np.zeros(geometry),
            )
            for time_constant, trap_density in sorted(
                zip(trap_time_constants, trap_densities)
            )
        ]

        self._trap_list = traps
        self._trapped_charge_array = np.zeros((len(traps), geometry[0], geometry[1]))
        self._trap_time_constants = np.array(trap_time_constants)
        self._trap_densities = np.array(trap_densities)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"{cls_name}<number_of_traps={len(self._trap_list)}>"

    @property
    def trap_list(self) -> Sequence[SimpleTrap]:
        """Return a list of traps."""
        return self._trap_list

    @property
    def trapped_charge_array(self) -> np.ndarray:
        """Return an array of trapped charge."""
        # for i, trap in enumerate(self.trap_list):
        #     out[i] = trap.charge
        return self._trapped_charge_array

    @trapped_charge_array.setter
    def trapped_charge_array(self, charge_3d: np.ndarray) -> None:
        """Set an array of trapped charge."""
        for i, trap in enumerate(self.trap_list):
            if not charge_3d[i].shape == trap.charge.shape:
                raise ValueError(
                    "Mismatch in shapes between saved trapped charge and input charge."
                )
            trap.charge = charge_3d[i]
        self._trapped_charge_array = charge_3d

    @property
    def trap_time_constants(self) -> np.ndarray:
        """Return trap time constants."""
        return self._trap_time_constants

    @property
    def trap_densities(self) -> np.ndarray:
        """Return trap densities."""
        return self._trap_densities

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""
import typing as t

import numpy as np

from pyxel.util.memory import get_size


# TODO: 'vg' should be the full volume and not the half
class Characteristics:
    """Characteristical attributes of the detector."""

    def __init__(
        self,
        quantum_efficiency: float = 1.0,  # unit: NA
        charge_to_volt_conversion: float = 1.0,  # unit: volt/electron
        pre_amplification: float = 1.0,  # unit: V/V
        analog_to_digital_gain: float = 1,  # unit: adu/V
        full_well_capacity: float = 0,  # unit: electron
    ):
        """Create an instance of `Characteristics`.

        Parameters
        ----------
        quantum_efficiency: float
            Quantum efficiency.
        charge_to_volt_conversion: float
            Sensitivity of charge readout. Unit: V/e-
        pre_amplification: float
            Gain of pre-amplifier. Unit: V/V
        analog_to_digital_gain: int
            Gain of the Analog-Digital Converter. Unit: ADU/V
        full_well_capacity: int
            Full well capacity. Unit: e-
        """
        if not (0.0 <= quantum_efficiency <= 1.0):
            raise ValueError("'qe' must be between 0.0 and 1.0.")

        if not (0.0 <= charge_to_volt_conversion <= 100.0):
            raise ValueError("'sv' must be between 0.0 and 100.0.")

        if not (0.0 <= pre_amplification <= 100.0):
            raise ValueError("'amp' must be between 0.0 and 100.0.")

        if analog_to_digital_gain not in range(65537):
            raise ValueError("'a2' must be between 0 and 65536.")

        if not (0.0 <= full_well_capacity <= 1.0e7):
            raise ValueError("'fwc' must be between 0 and 1e7.")

        self._quantum_efficiency = (
            quantum_efficiency
        )  # type: t.Union[float, np.ndarray]
        self._charge_to_volt_conversion = charge_to_volt_conversion
        self._pre_amplification = pre_amplification
        self._analog_to_digital_gain = analog_to_digital_gain
        self._full_well_capacity = full_well_capacity

        self._numbytes = 0

    @property
    def quantum_efficiency(self) -> t.Union[float, np.ndarray]:
        """Get Quantum efficiency."""
        return self._quantum_efficiency

    @quantum_efficiency.setter
    def quantum_efficiency(self, value: t.Union[float, np.ndarray]) -> None:
        """Set Quantum efficiency."""
        if np.min(value) < 0.0 and np.max(value) <= 1.0:
            raise ValueError("'QE' values must be between 0.0 and 1.0.")

        self._quantum_efficiency = value

    @property
    def charge_to_volt_conversion(self) -> float:
        """Get Sensitivity of charge readout."""
        return self._charge_to_volt_conversion

    @charge_to_volt_conversion.setter
    def charge_to_volt_conversion(self, value: float) -> None:
        """Set Sensitivity of charge readout."""
        if not (0.0 <= value <= 100.0):
            raise ValueError("'sv' must be between 0.0 and 100.0.")
        self._charge_to_volt_conversion = value

    @property
    def pre_amplification(self) -> float:
        """Get Gain of output amplifier."""
        return self._pre_amplification

    @pre_amplification.setter
    def pre_amplification(self, value: float) -> None:
        """Set Gain of output amplifier."""
        if not (0.0 <= value <= 100.0):
            raise ValueError("'amp' must be between 0.0 and 100.0.")

        self._pre_amplification = value

    @property
    def analog_to_digital_gain(self) -> float:
        """Get Gain of the Analog-Digital Converter."""
        return self._analog_to_digital_gain

    @analog_to_digital_gain.setter
    def analog_to_digital_gain(self, value: float) -> None:
        """Set Gain of the Analog-Digital Converter."""
        if value not in range(65537):
            raise ValueError("'a2' must be between 0 and 65536.")

        self._analog_to_digital_gain = value

    @property
    def full_well_capacity(self) -> float:
        """Get Full well capacity."""
        return self._full_well_capacity

    @full_well_capacity.setter
    def full_well_capacity(self, value: float) -> None:
        """Set Full well capacity."""
        if value not in range(10000001):
            raise ValueError("'fwc' must be between 0 and 1e+7.")

        self._full_well_capacity = value

    @property
    def numbytes(self) -> int:
        """Recursively calculates object size in bytes using Pympler library.

        Returns
        -------
        int
            Size of the object in bytes.
        """
        self._numbytes = get_size(self)
        return self._numbytes

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""
import numpy as np

from pyxel.util.memory import get_size
import typing as t


class Characteristics:
    """Characteristic attributes of the detector.

    Parameters
    ----------
    quantum_efficiency: float
        Quantum efficiency.
    charge_to_volt_conversion: float
        Sensitivity of charge readout. Unit: V/e-
    pre_amplification: float
        Gain of pre-amplifier. Unit: V/V
    adc_gain: float
        Gain of the Analog-Digital Converter. Unit: ADU/V
    full_well_capacity: float
        Full well capacity. Unit: e-
    adc_voltage_range: tuple of floats
        ADC voltage range.
    adc_bit_resolution: int
        ADC bit resolution.
    """

    def __init__(
        self,
        quantum_efficiency: float = 1,  # unit: NA
        charge_to_volt_conversion: float = 1.0e-6,  # unit: volt/electron
        pre_amplification: float = 1,  # unit: V/V
        adc_gain: float = 1,  # unit: adu/V
        full_well_capacity: float = 0,  # unit: electron
        adc_voltage_range: t.Tuple[float, float] = (0.0, 10.0),
        adc_bit_resolution: int = 8,
    ):
        if not (0.0 <= quantum_efficiency <= 1.0):
            raise ValueError("'quantum_efficiency' must be between 0.0 and 1.0.")
        if not (0.0 <= charge_to_volt_conversion <= 100.0):
            raise ValueError(
                "'charge_to_volt_conversion' must be between 0.0 and 100.0."
            )
        if not (0.0 <= pre_amplification <= 10000.0):
            raise ValueError("'pre_amplification' must be between 0.0 and 10000.0.")
        if adc_gain not in range(65537):
            raise ValueError("'adc_gain' must be between 0 and 65536.")
        if not (0.0 <= full_well_capacity <= 1.0e7):
            raise ValueError("'full_well_capacity' must be between 0 and 1e7.")
        if not (4 <= adc_bit_resolution <= 64):
            raise ValueError("'adc_bit_resolution' must be between 4 and 64.")
        if not len(adc_voltage_range) == 2:
            raise ValueError("Voltage range must have length of 2.")

        self._quantum_efficiency = quantum_efficiency  # type: float
        self._charge_to_volt_conversion = charge_to_volt_conversion  # type: float
        self._pre_amplification = pre_amplification  # type: float
        self._adc_gain = adc_gain  # type: float
        self._full_well_capacity = full_well_capacity  # type: float
        self._adc_voltage_range = adc_voltage_range  # type: t.Tuple[float, float]
        self._adc_bit_resolution = adc_bit_resolution  # type: int

        self._numbytes = 0

    @property
    def quantum_efficiency(self) -> float:
        """Get Quantum efficiency."""
        return self._quantum_efficiency

    @quantum_efficiency.setter
    def quantum_efficiency(self, value: float) -> None:
        """Set Quantum efficiency."""
        if np.min(value) < 0.0 and np.max(value) <= 1.0:
            raise ValueError("'quantum_efficiency' values must be between 0.0 and 1.0.")

        self._quantum_efficiency = value

    @property
    def charge_to_volt_conversion(self) -> float:
        """Get charge to volt conversion parameter."""
        return self._charge_to_volt_conversion

    @charge_to_volt_conversion.setter
    def charge_to_volt_conversion(self, value: float) -> None:
        """Set charge to volt conversion parameter."""
        if not (0.0 <= value <= 100.0):
            raise ValueError(
                "'charge_to_volt_conversion' must be between 0.0 and 100.0."
            )
        self._charge_to_volt_conversion = value

    @property
    def pre_amplification(self) -> float:
        """Get voltage pre-amplification gain."""
        return self._pre_amplification

    @pre_amplification.setter
    def pre_amplification(self, value: float) -> None:
        """Set voltage pre-amplification gain.."""
        if not (0.0 <= value <= 10000.0):
            raise ValueError("'pre_amplification' must be between 0.0 and 10000.0.")

        self._pre_amplification = value

    @property
    def adc_gain(self) -> float:
        """Get gain of the Analog-Digital Converter."""
        return self._adc_gain

    @adc_gain.setter
    def adc_gain(self, value: float) -> None:
        """Set gain of the Analog-Digital Converter."""
        if value not in range(65537):
            raise ValueError("'adc_gain' must be between 0 and 65536.")

        self._adc_gain = value

    @property
    def adc_bit_resolution(self) -> int:
        """Get bit resolution of the Analog-Digital Converter."""
        return self._adc_bit_resolution

    @adc_bit_resolution.setter
    def adc_bit_resolution(self, value: int) -> None:
        """Set bit resolution of the Analog-Digital Converter."""
        self._adc_bit_resolution = value

    @property
    def adc_voltage_range(self) -> t.Tuple[float, float]:
        """Get voltage range of the Analog-Digital Converter."""
        return self._adc_voltage_range

    @adc_voltage_range.setter
    def adc_voltage_range(self, value: t.Tuple[float, float]) -> None:
        """Set voltage range of the Analog-Digital Converter."""
        self._adc_voltage_range = value

    @property
    def full_well_capacity(self) -> float:
        """Get Full well capacity."""
        return self._full_well_capacity

    @full_well_capacity.setter
    def full_well_capacity(self, value: float) -> None:
        """Set Full well capacity."""
        if value not in range(10000001):
            raise ValueError("'full_well_capacity' must be between 0 and 1e+7.")

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

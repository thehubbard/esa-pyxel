#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from pyxel.util import get_size


class Characteristics:
    """Characteristic attributes of the detector.

    Parameters
    ----------
    quantum_efficiency : float, optional
        Quantum efficiency.
    charge_to_volt_conversion : float, optional
        Sensitivity of charge readout. Unit: V/e-
    pre_amplification : float, optional
        Gain of pre-amplifier. Unit: V/V
    full_well_capacity : float, optional
        Full well capacity. Unit: e-
    adc_voltage_range : tuple of floats, optional
        ADC voltage range. Unit: V
    adc_bit_resolution : int, optional
        ADC bit resolution.
    """

    def __init__(
        self,
        quantum_efficiency: float | None = None,  # unit: NA
        charge_to_volt_conversion: float | None = None,  # unit: volt/electron
        pre_amplification: float | None = None,  # unit: V/V
        full_well_capacity: float | None = None,  # unit: electron
        adc_bit_resolution: int | None = None,
        adc_voltage_range: tuple[float, float] | None = None,  # unit: V
    ):
        if quantum_efficiency is not None and not (0.0 <= quantum_efficiency <= 1.0):
            raise ValueError("'quantum_efficiency' must be between 0.0 and 1.0.")
        if charge_to_volt_conversion and not (
            0.0 <= charge_to_volt_conversion <= 100.0
        ):
            raise ValueError(
                "'charge_to_volt_conversion' must be between 0.0 and 100.0."
            )
        if pre_amplification is not None and not (0.0 <= pre_amplification <= 10_000.0):
            raise ValueError("'pre_amplification' must be between 0.0 and 10000.0.")
        if full_well_capacity is not None and not (0.0 <= full_well_capacity <= 1.0e7):
            raise ValueError("'full_well_capacity' must be between 0 and 1e7.")
        if adc_bit_resolution is not None and not (4 <= adc_bit_resolution <= 64):
            raise ValueError("'adc_bit_resolution' must be between 4 and 64.")

        if adc_voltage_range is not None:
            if not isinstance(adc_voltage_range, Sequence):
                raise TypeError("Voltage range must have length of 2.")

            if len(adc_voltage_range) != 2:
                raise ValueError("Voltage range must have length of 2.")

        self._quantum_efficiency = quantum_efficiency
        self._charge_to_volt_conversion = charge_to_volt_conversion
        self._pre_amplification = pre_amplification
        self._full_well_capacity = full_well_capacity

        if adc_voltage_range is None:
            volt_range: tuple[float, float] | None = None
        else:
            # Force 'volt_range' to be a tuple of 2 elements
            start_volt, end_volt = adc_voltage_range
            volt_range = (start_volt, end_volt)

        self._adc_voltage_range = volt_range
        self._adc_bit_resolution = adc_bit_resolution

        self._numbytes = 0

    def __eq__(self, other) -> bool:
        return (
            type(self) is type(other)
            and self._quantum_efficiency == other._quantum_efficiency
            and self._charge_to_volt_conversion == other._charge_to_volt_conversion
            and self._pre_amplification == other._pre_amplification
            and self._full_well_capacity == other._full_well_capacity
            and self._adc_voltage_range == other._adc_voltage_range
            and self._adc_bit_resolution == other._adc_bit_resolution
        )

    @property
    def quantum_efficiency(self) -> float:
        """Get Quantum efficiency."""
        if self._quantum_efficiency is None:
            raise ValueError(
                "'quantum_efficiency' not specified in detector characteristics."
            )

        return self._quantum_efficiency

    @quantum_efficiency.setter
    def quantum_efficiency(self, value: float) -> None:
        """Set Quantum efficiency."""
        # TODO: Refactor this
        if np.min(value) < 0.0 or np.max(value) > 1.0:
            raise ValueError("'quantum_efficiency' values must be between 0.0 and 1.0.")

        self._quantum_efficiency = value

    @property
    def charge_to_volt_conversion(self) -> float:
        """Get charge to volt conversion parameter."""
        if self._charge_to_volt_conversion is None:
            raise ValueError(
                "'charge_to_volt_conversion' not specified in detector characteristics."
            )

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
        if self._pre_amplification is None:
            raise ValueError(
                "'pre_amplification' not specified in detector characteristics."
            )

        return self._pre_amplification

    @pre_amplification.setter
    def pre_amplification(self, value: float) -> None:
        """Set voltage pre-amplification gain.."""
        if not (0.0 <= value <= 10_000.0):
            raise ValueError("'pre_amplification' must be between 0.0 and 10000.0.")

        self._pre_amplification = value

    @property
    def adc_bit_resolution(self) -> int:
        """Get bit resolution of the Analog-Digital Converter."""
        if self._adc_bit_resolution is None:
            raise ValueError(
                "'adc_bit_resolution' not specified in detector characteristics."
            )

        return self._adc_bit_resolution

    @adc_bit_resolution.setter
    def adc_bit_resolution(self, value: int) -> None:
        """Set bit resolution of the Analog-Digital Converter."""
        self._adc_bit_resolution = value

    @property
    def adc_voltage_range(self) -> tuple[float, float]:
        """Get voltage range of the Analog-Digital Converter."""
        if self._adc_voltage_range is None:
            raise ValueError(
                "'adc_voltage_range' not specified in detector characteristics."
            )

        return self._adc_voltage_range

    @adc_voltage_range.setter
    def adc_voltage_range(self, value: tuple[float, float]) -> None:
        """Set voltage range of the Analog-Digital Converter."""
        self._adc_voltage_range = value

    @property
    def full_well_capacity(self) -> float:
        """Get Full well capacity."""
        if self._full_well_capacity is None:
            raise ValueError(
                "'full_well_capacity' not specified in detector characteristics."
            )

        return self._full_well_capacity

    @full_well_capacity.setter
    def full_well_capacity(self, value: float) -> None:
        """Set Full well capacity."""
        if not (0 <= value <= 10_000_000):
            raise ValueError("'full_well_capacity' must be between 0 and 1e+7.")

        self._full_well_capacity = value

    @property
    def system_gain(self) -> float:
        """Get system gain."""
        return (
            self.quantum_efficiency
            * self.pre_amplification
            * self.charge_to_volt_conversion
            * 2**self.adc_bit_resolution
        ) / (max(self.adc_voltage_range) - min(self.adc_voltage_range))

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

    def to_dict(self) -> Mapping:
        """Get the attributes of this instance as a `dict`."""
        return {
            "quantum_efficiency": self._quantum_efficiency,
            "charge_to_volt_conversion": self._charge_to_volt_conversion,
            "pre_amplification": self._pre_amplification,
            "full_well_capacity": self._full_well_capacity,
            "adc_bit_resolution": self._adc_bit_resolution,
            "adc_voltage_range": self._adc_voltage_range,
        }

    @classmethod
    def from_dict(cls, dct: Mapping):
        """Create a new instance from a `dict`."""
        # Late import to speedup start-up time
        from toolz import dicttoolz

        # TODO: This is a simplistic implementation. Improve this.
        # Extract param 'adc_voltage_range'
        param: Iterable[float] | None = dct.get("adc_voltage_range")
        new_dct: Mapping = dicttoolz.dissoc(dct, "adc_voltage_range")

        if param is None:
            adc_voltage_range: tuple[float, float] | None = None
        else:
            adc_voltage_min, adc_voltage_max = tuple(param)
            adc_voltage_range = adc_voltage_min, adc_voltage_max

        return cls(adc_voltage_range=adc_voltage_range, **new_dct)

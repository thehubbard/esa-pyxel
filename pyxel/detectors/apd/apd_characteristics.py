#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""
from pyxel.detectors.characteristics import Characteristics
import typing as t
import numpy as np


class APDCharacteristics(Characteristics):
    """Characteristic attributes of the detector.

    Parameters
    ----------
    quantum_efficiency: float, optional
        Quantum efficiency.
    charge_to_volt_conversion: float, optional
        Sensitivity of charge readout. Unit: V/e-
    pre_amplification: float, optional
        Gain of pre-amplifier. Unit: V/V
    full_well_capacity: float, optional
        Full well capacity. Unit: e-
    adc_voltage_range: tuple of floats, optional
        ADC voltage range. Unit: V
    adc_bit_resolution: int, optional
        ADC bit resolution.
    apd_gain: float, optional
        APD gain. Unit: electron/electron
    """

    def __init__(
        self,
        quantum_efficiency: t.Optional[float] = None,  # unit: NA
        charge_to_volt_conversion: t.Optional[float] = None,  # unit: volt/electron
        pre_amplification: t.Optional[float] = None,  # unit: V/V
        full_well_capacity: t.Optional[float] = None,  # unit: electron
        adc_bit_resolution: t.Optional[int] = None,
        adc_voltage_range: t.Optional[t.Tuple[float, float]] = None,  # unit: V
        apd_gain: t.Optional[float] = None,  # unit: electron/electron
    ):
        super().__init__(
            quantum_efficiency=quantum_efficiency,
            charge_to_volt_conversion=charge_to_volt_conversion,
            pre_amplification=pre_amplification,
            full_well_capacity=full_well_capacity,
            adc_voltage_range=adc_voltage_range,
            adc_bit_resolution=adc_bit_resolution,
        )

        if apd_gain and not (1.0 <= apd_gain <= 1000.0):
            raise ValueError("'apd_gain' must be between 1.0 and 1000.0.")

        self._apd_gain = apd_gain

    @property
    def apd_gain(self) -> float:
        """Get APD gain."""
        if self._apd_gain:
            return self._apd_gain
        else:
            raise ValueError("'apd_gain' not specified in detector characteristics.")

    @apd_gain.setter
    def apd_gain(self, value: float) -> None:
        """Set APD gain."""
        if np.min(value) < 1.0 and np.max(value) <= 1000.0:
            raise ValueError("'apd_gain' values must be between 1.0 and 1000.")

        self._apd_gain = value

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""
import typing as t
import numpy as np
import math
import astropy.constants as const


class APDCharacteristics:
    """Characteristic attributes of the detector.

    Parameters
    ----------
    quantum_efficiency: float, optional
        Quantum efficiency.
    full_well_capacity: float, optional
        Full well capacity. Unit: e-
    adc_bit_resolution: int, optional
        ADC bit resolution.
    adc_voltage_range: tuple of floats, optional
        ADC voltage range. Unit: V
    avalanche_gain: float, optional
        APD gain. Unit: electron/electron
    pixel_reset_voltage
    common_voltage
    roic_gain
    """

    def __init__(
        self,
        quantum_efficiency: t.Optional[float] = None,  # unit: NA
        full_well_capacity: t.Optional[float] = None,  # unit: electron
        adc_bit_resolution: t.Optional[int] = None,
        adc_voltage_range: t.Optional[t.Tuple[float, float]] = None,  # unit: V
        avalanche_gain: t.Optional[float] = None,  # unit: electron/electron
        pixel_reset_voltage: t.Optional[float] = None,  # unit: V
        common_voltage: t.Optional[float] = None,  # unit: V
        roic_gain: t.Optional[float] = None,  # unit: V
    ):

        self._avalanche_gain = avalanche_gain
        self._common_voltage = common_voltage
        self._pixel_reset_voltage = pixel_reset_voltage

        if avalanche_gain and pixel_reset_voltage and common_voltage:
            raise ValueError(
                "Please only specify two inputs out of: avalanche gain, pixel reset voltage, common voltage."
            )
        elif avalanche_gain and pixel_reset_voltage and not common_voltage:
            self._avalanche_bias = self.gain_to_bias_saphira(avalanche_gain)
            self._common_voltage = pixel_reset_voltage - self.avalanche_bias
        elif avalanche_gain and common_voltage and not pixel_reset_voltage:
            self._avalanche_bias = self.gain_to_bias_saphira(avalanche_gain)
            self._pixel_reset_voltage = common_voltage + self.avalanche_bias
        elif common_voltage and pixel_reset_voltage and not avalanche_gain:
            self._avalanche_bias = pixel_reset_voltage - common_voltage
            self._avalanche_gain = self.bias_to_gain_saphira(self.avalanche_bias)
        else:
            raise ValueError(
                "Not enough input parameters provided to calculate avalanche bias!"
            )

        if quantum_efficiency and not (0.0 <= quantum_efficiency <= 1.0):
            raise ValueError("'quantum_efficiency' must be between 0.0 and 1.0.")
        if avalanche_gain and not (1.0 <= avalanche_gain <= 1000.0):
            raise ValueError("'apd_gain' must be between 1.0 and 1000.0.")
        if adc_bit_resolution and not (4 <= adc_bit_resolution <= 64):
            raise ValueError("'adc_bit_resolution' must be between 4 and 64.")
        if adc_voltage_range and not len(adc_voltage_range) == 2:
            raise ValueError("Voltage range must have length of 2.")
        if full_well_capacity and not (0.0 <= full_well_capacity <= 1.0e7):
            raise ValueError("'full_well_capacity' must be between 0 and 1e7.")

        self._quantum_efficiency = quantum_efficiency
        self._full_well_capacity = full_well_capacity
        self._adc_voltage_range = adc_voltage_range
        self._adc_bit_resolution = adc_bit_resolution
        self._node_capacitance = self.bias_to_node_capacitance_saphira(
            self.avalanche_bias
        )
        self._roic_gain = roic_gain
        self._charge_to_volt_conversion = self.detector_gain_saphira(
            capacitance=self._node_capacitance, roic_gain=roic_gain
        )

    @property
    def avalanche_gain(self) -> float:
        """Get APD gain."""
        if self._avalanche_gain:
            return self._avalanche_gain
        else:
            raise ValueError("'apd_gain' not specified in detector characteristics.")

    @avalanche_gain.setter
    def avalanche_gain(self, value: float) -> None:
        """Set APD gain."""
        if np.min(value) < 1.0 and np.max(value) <= 1000.0:
            raise ValueError("'apd_gain' values must be between 1.0 and 1000.")
        self._avalanche_gain = value
        self._avalanche_bias = self.gain_to_bias_saphira(value)
        self._common_voltage = self.pixel_reset_voltage - self.avalanche_bias

    @property
    def pixel_reset_voltage(self) -> float:
        """Get APD gain."""
        if self._pixel_reset_voltage:
            return self._pixel_reset_voltage
        else:
            raise ValueError("'apd_gain' not specified in detector characteristics.")

    @pixel_reset_voltage.setter
    def pixel_reset_voltage(self, value: float) -> None:
        """Set APD gain."""
        self._avalanche_bias = value - self._common_voltage
        self._avalanche_gain = self.bias_to_gain_saphira(self._avalanche_bias)
        self._pixel_reset_voltage = value

    @property
    def common_voltage(self) -> float:
        """Get APD gain."""
        if self._common_voltage:
            return self._common_voltage
        else:
            raise ValueError("'apd_gain' not specified in detector characteristics.")

    @common_voltage.setter
    def common_voltage(self, value: float) -> None:
        """Set APD gain."""
        self._avalanche_bias = self.pixel_reset_voltage - value
        self._avalanche_gain = self.bias_to_gain_saphira(self._avalanche_bias)
        self._common_voltage = value

    @property
    def avalanche_bias(self) -> float:
        """Get APD gain."""
        if self._avalanche_bias:
            return self._avalanche_bias
        else:
            raise ValueError("'apd_gain' not specified in detector characteristics.")

    @property
    def roic_gain(self) -> float:
        """Get APD gain."""
        if self._roic_gain:
            return self._roic_gain
        else:
            raise ValueError("'roic_gain' not specified in detector characteristics.")

    @property
    def node_capacitance(self) -> float:
        """Get APD gain."""
        self._node_capacitance = self.bias_to_node_capacitance_saphira(
            self.avalanche_bias
        )
        return self._node_capacitance

    @property
    def charge_to_volt_conversion(self):
        self._charge_to_volt_conversion = self.detector_gain_saphira(
            capacitance=self.node_capacitance, roic_gain=self.roic_gain
        )
        return self._charge_to_volt_conversion

    @staticmethod
    def bias_to_node_capacitance_saphira(bias: float) -> float:
        """

        Parameters
        ----------
        bias

        Returns
        -------

        """
        if bias < 1:
            raise ValueError(
                "Warning! Node capacitance calculation is inaccurate for bias voltages <1 V!"
            )

        # From [2] (Mk13 ME1000; data supplied by Leonardo):
        bias = [1, 1.5, 2.5, 3.5, 4.5, 6.5, 8.5, 10.5]
        capacitance = [46.5, 41.3, 37.3, 34.8, 33.2, 31.4, 30.7, 30.4]

        output_capacitance = np.interp(bias, bias, capacitance)

        return output_capacitance * 1.0e-15

    @staticmethod
    def bias_to_gain_saphira(bias: float) -> float:
        """The formula ignores the soft knee between the linear and
        unity gain ranges, but should be close enough. [2] (Mk13 ME1000)"""

        gain = 2 ** ((bias - 2.65) / 2.17)

        if gain < 1.0:
            gain = 1.0  # Unity gain is lowest

        return gain

    @staticmethod
    def gain_to_bias_saphira(gain: float) -> float:
        """The formula ignores the soft knee between the linear and
        unity gain ranges, but should be close enough. [2] (Mk13 ME1000)"""

        bias = (2.17 * math.log2(gain)) + 2.65

        return bias

    @staticmethod
    def detector_gain_saphira(capacitance: float, roic_gain: float) -> float:
        """"""
        return roic_gain * (const.e.value / capacitance)

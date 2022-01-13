#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""
from pyxel.detectors import Characteristics


class CMOSCharacteristics(Characteristics):
    """Characteristical attributes of a CMOS-based detector."""

    # def __init__(
    #     self,
    #     # Parameters for `Characteristics`
    #     quantum_efficiency: float = 1.0,
    #     charge_to_volt_conversion: float = 1.0,
    #     pre_amplification: float = 1.0,
    #     adc_gain: float = 1.0,
    #     full_well_capacity: float = 0,
    #     # Parameters specific `CMOSCharacteristics`
    #     cutoff_wavelength: float = 2.5,  # unit: um
    #     vbiaspower: float = 3.35,  # unit: V
    #     dsub: float = 0.5,  # unit: V
    #     vreset: float = 0.25,  # unit: V
    #     biasgate: float = 2.3,  # unit: V
    #     preampref: float = 1.7,  # unit: V
    # ):
    #     """Create an instance of `CMOSCharacteristics`.
    #
    #     Parameters
    #     ----------
    #     cutoff_wavelength: float
    #         Cutoff wavelength. Unit: um
    #     vbiaspower: float
    #         VBIASPOWER. Unit: V
    #     dsub: float
    #         DSUB. Unit: V
    #     vreset: float
    #         VRESET. Unit: V
    #     biasgate: float
    #         BIASGATE. Unit: V
    #     preampref: float
    #         PREAMPREF. Unit: V
    #     """
    #
    #
    #     if not (1.7 <= cutoff_wavelength <= 15.0):
    #         raise ValueError("'cutoff' must be between 1.7 and 15.0.")
    #
    #     if not (0.0 <= vbiaspower <= 3.4):
    #         raise ValueError("'vbiaspower' must be between 0.0 and 3.4.")
    #
    #     if not (0.3 <= dsub <= 1.0):
    #         raise ValueError("'dsub' must be between 0.3 and 1.0.")
    #
    #     if not (0.0 <= vreset <= 0.3):
    #         raise ValueError("'vreset' must be between 0.0 and 0.3.")
    #
    #     if not (1.8 <= biasgate <= 2.6):
    #         raise ValueError("'biasgate' must be between 1.8 and 2.6.")
    #
    #     if not (0.0 <= preampref <= 4.0):
    #         raise ValueError("'preampref' must be between 0.0 and 4.0.")
    #
    #     super().__init__(
    #         quantum_efficiency=quantum_efficiency,
    #         charge_to_volt_conversion=charge_to_volt_conversion,
    #         pre_amplification=pre_amplification,
    #         adc_gain=adc_gain,
    #         full_well_capacity=full_well_capacity,
    #     )
    #     self._cutoff_wavelength = cutoff_wavelength
    #     # self._vbiaspower = vbiaspower
    #     # self._dsub = dsub
    #     # self._vreset = vreset
    #     # self._biasgate = biasgate
    #     # self._preampref = preampref
    #
    # @property
    # def cutoff_wavelength(self) -> float:
    #     """Get Cutoff wavelength."""
    #     return self._cutoff
    #
    # @cutoff_wavelength.setter
    # def cutoff_wavelength(self, value: float) -> None:
    #     """Set Cutoff wavelength."""
    #     if not (1.7 <= value <= 15.0):
    #         raise ValueError("'cutoff' must be between 1.7 and 15.0.")
    #
    #     self._cutoff = value
    #
    # @property
    # def vbiaspower(self) -> float:
    #     """Get VBIASPOWER."""
    #     return self._vbiaspower
    #
    # @vbiaspower.setter
    # def vbiaspower(self, value: float) -> None:
    #     """Set VBIASPOWER."""
    #     if not (0.0 <= value <= 3.4):
    #         raise ValueError("'vbiaspower' must be between 0.0 and 3.4.")
    #
    #     self._vbiaspower = value
    #
    # @property
    # def dsub(self) -> float:
    #     """Get DSUB."""
    #     return self._dsub
    #
    # @dsub.setter
    # def dsub(self, value: float) -> None:
    #     """Set DSUB."""
    #     if not (0.3 <= value <= 1.0):
    #         raise ValueError("'dsub' must be between 0.3 and 1.0.")
    #
    #     self._dsub = value
    #
    # @property
    # def vreset(self) -> float:
    #     """Get VREST."""
    #     return self._vreset
    #
    # @vreset.setter
    # def vreset(self, value: float) -> None:
    #     """Set VRESET."""
    #     if not (0.0 <= value <= 0.3):
    #         raise ValueError("'vreset' must be between 0.0 and 0.3.")
    #
    #     self._vreset = value
    #
    # @property
    # def biasgate(self) -> float:
    #     """Get BIASGATE."""
    #     return self._biasgate
    #
    # @biasgate.setter
    # def biasgate(self, value: float) -> None:
    #     """Set BIASGATE."""
    #     if not (1.8 <= value <= 2.6):
    #         raise ValueError("'biasgate' must be between 1.8 and 2.6.")
    #
    #     self._biasgate = value
    #
    # @property
    # def preampref(self) -> float:
    #     """Get PREAMPREF."""
    #     return self._preampref
    #
    # @preampref.setter
    # def preampref(self, value: float) -> None:
    #     """Set PREAMPREF."""
    #     if not (0.0 <= value <= 4.0):
    #         raise ValueError("'preampref' must be between 0.0 and 4.0.")
    #
    #     self._preampref = value

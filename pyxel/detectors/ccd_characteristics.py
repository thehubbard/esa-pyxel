#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""
from pyxel.detectors import Characteristics


class CCDCharacteristics(Characteristics):
    """Characteristical attributes of a CCD detector."""

    # def __init__(
    #     self,
    #     # Parameters for `Characteristics`
    #     quantum_efficiency: float = 1.0,  # unit: NA
    #     charge_to_volt_conversion: float = 1.0,  # unit: volt/electron
    #     pre_amplification: float = 1.0,  # unit: V/V
    #     adc_gain: int = 1,  # unit: adu/V
    #     full_well_capacity: int = 0,  # unit: electron
    # ):
    #     """Create an instance of `CCDCharacteristics`."""
    #     super().__init__(
    #         quantum_efficiency=quantum_efficiency,
    #         charge_to_volt_conversion=charge_to_volt_conversion,
    #         pre_amplification=pre_amplification,
    #         adc_gain=adc_gain,
    #         full_well_capacity=full_well_capacity,
    #     )
    #
    # def __repr__(self) -> str:
    #     cls_name = self.__class__.__name__  # type: str
    #     return f"{cls_name}"

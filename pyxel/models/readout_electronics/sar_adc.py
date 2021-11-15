#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""SAR ADC model."""
import logging

import numpy as np

from pyxel.detectors import Detector


def sar_adc(detector: Detector, adc_bits: int = 16, range_volt: tuple = (0, 5)) -> None:
    """Digitize signal array using SAR ADC logic.

    :param detector: Pyxel Detector object
    :param adc_bits: integer: Number of bits for the ADC
    :param range_volt: tuple with min anx max volt value
    """
    # import numpy as np
    logging.info("")
    # Extract the data to digitize
    data = detector.signal.array
    data_digitized = np.zeros((detector.geometry.row, detector.geometry.col))
    # First normalize the data to voltage since there is no model for
    # the conversion photogenerated carrier > volts in the model/charge_measure
    data = data * range_volt[1] / np.max(data)
    # steps = list()
    # Set the reference voltage of the ADC to half the max
    ref = range_volt[1] / 2.0
    # For each bits, compare the value of the ref to the capacitance value
    for i in np.arange(0, adc_bits):
        # digital value associated with this step
        digital_value = 2 ** (adc_bits - (i + 1))
        # All data that is higher than the ref is equal to the dig. value
        data_digitized[data >= ref] += digital_value
        # Subtract ref value from the data
        data[data >= ref] -= ref
        # steps.append(ref)
        # Divide reference voltage by 2 for next step
        ref /= 2.0

    detector.image.array = data_digitized
    # return data_digitized, steps

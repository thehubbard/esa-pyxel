#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
import numpy as np
from pyxel.detectors import Detector


def apply_simple_adc(
    signal: np.ndarray, bit_resolution: int, voltage_range: t.Tuple[float, float]
) -> np.ndarray:
    """Apply digitization.

    Parameters
    ----------
    signal: np.ndarray
        Input signal.
    bit_resolution: int
        ADC bit resolution.
    voltage_range: tuple of floats
        ADC voltage range.

    Returns
    -------
    output: ndarray
    """
    bins = np.linspace(
        start=voltage_range[0], stop=voltage_range[1], num=2 ** bit_resolution
    )
    output = np.digitize(x=signal, bins=bins[:-1], right=True)


def simple_adc(
    detector: Detector,
    bit_resolution: t.Optional[int] = None,
    voltage_range: t.Optional[t.Tuple[float, float]] = None,
) -> None:
    """Apply simple Analog to Digital conversion

    Parameters
    ----------
    detector: Detector
    bit_resolution: int, optional
    voltage_range: tuple of floats, optional
    """
    if bit_resolution is None:
        final_bit_resolution = detector.characteristics.adc_bit_resolution
    else:
        final_bit_resolution = bit_resolution
    if voltage_range is None:
        final_voltage_range = detector.characteristics.adc_voltage_range
    else:
        final_voltage_range = voltage_range

    if not (4 <= final_bit_resolution <= 64):
        raise ValueError("'adc_bit_resolution' must be between 4 and 64.")
    if not len(final_voltage_range) == 2:
        raise ValueError("Voltage range must have length of 2.")

    detector.image.array = apply_simple_adc(
        signal=detector.signal.array,
        bit_resolution=final_bit_resolution,
        voltage_range=final_voltage_range,
    )

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
"""Simple ADC model functions."""

import typing as t

import numpy as np
from typing_extensions import Literal

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
    return output


def simple_adc(
    detector: Detector,
    bit_resolution: t.Optional[int] = None,
    voltage_range: t.Optional[t.Tuple[float, float]] = None,
    data_type: Literal["uint16", "uint32", "uint64", "uint"] = "uint32",
) -> None:
    """Apply simple Analog to Digital conversion.

    Parameters
    ----------
    detector: Detector
        Pyxel Detector object.
    bit_resolution: int, optional
        ADC bit resolution.
    voltage_range: tuple of floats, optional
        ADC voltage range.
    data_type : str
        The desired data-type for the Image array. The data-type must be an unsigned integer.
        Valid values: 'uint16', 'uint32', 'uint64', 'uint'
        Invalid values: 'int16', 'int32', 'int64', 'int', 'float'...
    """
    if bit_resolution is None:
        final_bit_resolution = detector.characteristics.adc_bit_resolution
    else:
        final_bit_resolution = bit_resolution
    if voltage_range is None:
        final_voltage_range = detector.characteristics.adc_voltage_range
    else:
        final_voltage_range = voltage_range

    try:
        d_type = np.dtype(data_type)  # type: np.dtype
    except TypeError as ex:
        raise TypeError(
            "Can not locate the type defined as `data_type` argument in yaml file."
        ) from ex

    if not issubclass(d_type.type, np.integer):
        raise TypeError("Expecting a signed/unsigned integer.")

    if not (4 <= final_bit_resolution <= 64):
        raise ValueError("'adc_bit_resolution' must be between 4 and 64.")
    if not len(final_voltage_range) == 2:
        raise ValueError("Voltage range must have length of 2.")

    detector.image.array = np.asarray(
        apply_simple_adc(
            signal=detector.signal.array,
            bit_resolution=final_bit_resolution,
            voltage_range=final_voltage_range,
        ),
        dtype=d_type,
    )

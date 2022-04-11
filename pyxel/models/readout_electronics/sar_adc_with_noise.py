#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

""":term:`SAR` :term:`ADC` model with noise."""

import typing as t

import numpy as np

from pyxel.detectors import Detector


def apply_sar_adc_with_noise(
    signal_2d: np.ndarray,
    num_rows: int,
    num_cols: int,
    strengths: np.ndarray,
    noises: np.ndarray,
    max_volt: float,
    adc_bits: int,
) -> np.ndarray:
    """Apply :term:`SAR` :term:`ADC` with noise.

    Parameters
    ----------
    signal_2d : ndarray
    num_rows : int
    num_cols : int
    strengths : ndarray
    noises : ndarray
    max_volt : float
        Max volt of the ADC.
    adc_bits : int
        Number of bits the value will be encoded with.

    Returns
    -------
    ndarray
        2D digitized array.
    """
    data_digitized_2d = np.zeros((num_rows, num_cols))

    # First normalize the data to voltage since there is no model for
    # the conversion photogenerated carrier > volts in the model/charge_measure
    # signal_normalized_2d = signal_2d * max_volt / np.max(signal_2d)
    signal_normalized_2d = signal_2d.copy()

    # Set the reference voltage of the ADC to half the max
    ref_2d = np.full(shape=(num_rows, num_cols), fill_value=max_volt / 2.0)

    # For each bits, compare the value of the ref to the capacitance value
    for i in np.arange(adc_bits):
        strength = strengths[i]
        noise = noises[i]

        # digital value associated with this step
        digital_value = 2 ** (adc_bits - (i + 1))

        ref_2d += np.random.normal(loc=strength, scale=noise, size=(num_rows, num_cols))

        # All data that is higher than the ref is equal to the dig. value
        mask_2d = signal_normalized_2d >= ref_2d
        data_digitized_2d[mask_2d] += digital_value

        # Subtract ref value from the data
        signal_normalized_2d[mask_2d] -= ref_2d

        # Divide reference voltage by 2 for next step
        ref_2d /= 2.0

    return data_digitized_2d


# TODO: documentation, range volt - only max is used
def sar_adc_with_noise(
    detector: Detector,
    strengths: t.Tuple[float, ...],
    noises: t.Tuple[float, ...],
) -> None:
    """Digitize signal array using :term:`SAR` (Successive Approximation Register) :term:`ADC` logic with noise.

    Successive-approximation-register (SAR) analog-to-digital converters (ADCs)
    for each bit, will compare the randomly perturbated reference voltage to the voltage
    of the pixel, adding the corresponding digital value if it is superior.
    The perturbations are generated randomly for each pixel.

    Strength and noise are the parameters that regulate the random fluctuations of
    the reference voltage.

    Parameters
    ----------
    detector : Detector
        Pyxel Detector object.
    strengths : tuple of float
    noises : tuple of float
    """
    assert len(strengths) == len(noises) == detector.characteristics.adc_bit_resolution

    min_volt, max_volt = detector.characteristics.adc_voltage_range
    adc_bits = detector.characteristics.adc_bit_resolution

    image_2d = apply_sar_adc_with_noise(
        signal_2d=detector.signal.array,
        num_rows=detector.geometry.row,
        num_cols=detector.geometry.col,
        strengths=strengths,
        noises=noises,
        max_volt=max_volt,
        adc_bits=adc_bits,
    )  # type: np.ndarray

    detector.image.array = image_2d

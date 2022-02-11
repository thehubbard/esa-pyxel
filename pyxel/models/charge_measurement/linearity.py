#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Linearity models."""
import typing as t

import numpy as np

from pyxel.detectors import Detector


def compute_poly_linearity(
    array_2d: np.ndarray,
    coefficients: t.Sequence[float],
) -> np.ndarray:
    """Add non-linearity to an array of values following a polynomial function.

    Parameters
    ----------
    array_2d: ndarray
        Input array.
    coefficients: list of float
        Coefficients of the polynomial function.

    Returns
    -------
    signal: np.ndarray
    """

    polynomial_function = np.polynomial.polynomial.Polynomial(coefficients)

    non_linear_signal = polynomial_function(array_2d)

    return non_linear_signal


def output_node_linearity_poly(
    detector: Detector,
    coefficients: t.Sequence[float],
) -> None:
    """Add non-linearity to signal array to simulate the non-linearity of the output node circuit.

    The non-linearity is simulated by a polynomial function. The user specifies the polynomial coefficients.

    detector Signal unit: Volt

    Parameters
    ----------
    detector: Detector
        Pyxel Detector object.
    coefficients: list of float
        Coefficient of the polynomial function.
    """
    if len(coefficients) == 0:
        raise ValueError("Length of coefficient list should be more than 0.")

    signal_mean_array = detector.signal.array.astype("float64")
    signal_non_linear = compute_poly_linearity(
        array_2d=signal_mean_array, coefficients=coefficients
    )

    if np.any(signal_non_linear < 0):
        raise ValueError(
            "Signal array contains negative values after applying non-linearity model!"
        )

    detector.signal.array = signal_non_linear

# ---------------------------------------------------------------------
def non_linear_model_v2(detector: Detector) -> None:
    """

    :return:
    """
    # Fixed capacitance
    fixed_capa = detector.characteristics._fixed_capacitance*1E-15 # in F
    # Surface of the diode, assumed to be planar and circular
    Ad = np.pi * (detector.geometry._implementation_diameter/ 2.0 * 1e-6) ** 2
    # Detector bias
    vb = detector.characteristics._vreset - detector.characteristics._dsub
    # Build in potential
    vbi = detector.material._vbi_build_in_potential

    # Initial value  of the diode capacitance
    co = Ad * np.sqrt((Q * detector.material.eps * EPS0) / (2 * (vbi - vb)) * \
                      ((detector.material.n_acceptor * 1e6 * detector.material.n_donor * 1e6) / (
                                  detector.material.n_acceptor * 1e6 + detector.material.n_donor * 1e6)))

    # Resolution of 2nd order equation
    ao = 2. * co * vbi * np.sqrt(1. - vb / vbi) - fixed_capa * vb - detector.pixel.array*Q
    b = -2. * co / fixed_capa
    c = ao / (fixed_capa * vbi) + 1.
    discriminant = b**2 - 4.*c*(-1.)

    u1 = (-b - np.sqrt(discriminant))/2.
    v1 = (1. - u1**2)*vbi - vb # vb is substracted it only deals with offset level

    array = np.copy(v1)  # unit if V, voltage at the gate of the pixel SFD
    detector.signal.array = array.astype("float64")

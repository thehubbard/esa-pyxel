#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Linearity models."""
import typing as t

import matplotlib.pyplot as plt
import numpy as np

from pyxel.detectors import Detector
from pyxel.models.charge_measurement.non_linearity_calculation import euler

# Universal global constants
M_ELECTRON = 9.10938356e-31  # kg     #TODO: put these global constants to a data file
KB = 1.38064852e-23  #
Q = 1.60217662e-19
EPS0 = 8.85418782e-12


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
    polynomial_function = np.polynomial.polynomial.Polynomial(
        [0, 1.6 * 1e-19 / 4.3e-14]
    )

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

    signal_mean_array = detector.charge.array.astype("float64")
    signal_non_linear = compute_poly_linearity(
        array_2d=signal_mean_array, coefficients=coefficients
    )

    if np.any(signal_non_linear < 0):
        raise ValueError(
            "Signal array contains negative values after applying non-linearity model!"
        )

    detector.signal.array = signal_non_linear


# ---------------------------------------------------------------------
def compute_physical_non_linearity_model_1(
    array_2d: np.ndarray,
    temperature: float,  # Detector operating temperature
    vbias: float,
) -> np.ndarray:
    """
    Parameters
    ----------
    detector

    Returns
    -------
    """
    # Derivation of Cd concentration in the alloy,  it depends on cutoff wavelength and targeted operating temperature
    # Here we are considering the case where the detector is operated at its nominal temperature, it might not be always the case
    cutoff = 2.1
    Eg_targeted = 1.24 / cutoff  # cutoff is um and Eg in eV
    xcd = np.linspace(0.2, 0.6, 1000)
    targeted_operating_temperature = temperature
    Eg_calculated = hgcdte_bandgap(
        xcd, targeted_operating_temperature
    )  # Expected bandgap
    index = np.where(Eg_calculated > Eg_targeted)[0][0]
    x_cd = xcd[index]  # Targeted cadmium concentration in the HgCdTe alloy
    # Calculate the effective bandgap value at the temperature at which simulations are performed.
    Eg = hgcdte_bandgap(x_cd, temperature)

    # Acceptor and donor doping concentrations
    n_acceptor = 1e18  # in atoms/cm3
    n_donor = 3e15  # in atoms/cm3

    # Intrinsic carrier concentration
    # Standard semiconductor expression can also be used
    ni = (
        (
            5.585
            - 3.820 * x_cd
            + 1.753 * 1e-3 * temperature
            - 1.364 * 1e-3 * temperature * x_cd
        )
        * 1e14
        * Eg ** 0.75
        * temperature ** 1.5
        * np.exp(-Q * Eg / (2 * KB * temperature))
    )

    # Build in potential
    vbi = KB * temperature / Q * np.log(n_acceptor * n_donor / ni ** 2)

    # HgCdTe dielectric constant
    eps = 20.5 - 15.6 * x_cd + 5.7 * x_cd ** 2

    # Surface of the diode, assumed to be planar
    diode_diameter = 10  # in um
    Ad = (
        np.pi * (diode_diameter / 2.0 * 1e-6) ** 2
    )  # Surface of the diode, assumed to be circular

    # Initial value  of the diode capacitance
    co = Ad * np.sqrt(
        (Q * eps * EPS0)
        / (2 * (vbi - vbias))
        * ((n_acceptor * 1e6 * n_donor * 1e6) / (n_acceptor * 1e6 + n_donor * 1e6))
    )

    non_linear_signal = (
        1
        / (2 * vbi)
        * (Q * array_2d / co) ** 2
        * (-1 + np.sqrt(1 + 4 * (co / (Q * array_2d)) ** 2 * vbi ** 2))
    )

    return non_linear_signal


def output_node_linearity_non_linearity_model_1(
    detector: Detector,
) -> None:

    signal_mean_array = detector.charge.array.astype("float64")
    signal_non_linear = compute_physical_non_linearity_model_1(
        array_2d=signal_mean_array,
        temperature=detector.environment.temperature,
        vbias=-0.250,
    )

    detector.signal.array = signal_non_linear


# ---------------------------------------------------------------------
def compute_physical_non_linearity_model_2(
    array_2d: np.ndarray,
    temperature: float,  # Detector operating temperature
    vbias: float,
    fixed_capa: float,  # Additionnal fixed capacitance
) -> np.ndarray:
    """
    Parameters
    ----------
    detector

    Returns
    -------

    """
    # Derivation of Cd concentration in the alloy,  it depends on cutoff wavelength and targeted operating temperature
    # Here we are considering the case where the detector is operated at its nominal temperature, it might not be always the case
    cutoff = 2.1  # Can be extracted from CMOS characteristics ?
    Eg_targeted = 1.24 / cutoff  # cutoff is um and Eg in eV
    xcd = np.linspace(0.2, 0.6, 1000)
    targeted_operating_temperature = temperature
    Eg_calculated = hgcdte_bandgap(
        xcd, targeted_operating_temperature
    )  # Expected band-gap
    index = np.where(Eg_calculated > Eg_targeted)[0][0]
    x_cd = xcd[index]  # Targeted cadmium concentration in the HgCdTe alloy

    # Calculate the effective band-gap value at the temperature at which simulations are performed.
    Eg = hgcdte_bandgap(x_cd, temperature)

    # Acceptor and donor doping concentrations
    n_acceptor = 1e18  # in atom/cm3
    n_donor = 3e15  # in atoms/cm3

    # Intrinsic carrier concentration
    # Standard semiconductor expression can also be used
    ni = (
        (
            5.585
            - 3.820 * x_cd
            + 1.753 * 1e-3 * temperature
            - 1.364 * 1e-3 * temperature * x_cd
        )
        * 1e14
        * Eg ** 0.75
        * temperature ** 1.5
        * np.exp(-Q * Eg / (2 * KB * temperature))
    )  # in carriers/cm3

    # Build in potential
    vbi = KB * temperature / Q * np.log(n_acceptor * n_donor / ni ** 2)  # in V

    # HgCdTe dielectric constant
    eps = 20.5 - 15.6 * x_cd + 5.7 * x_cd ** 2  # without dimension

    # Surface of the diode, assumed to be planar
    diode_diameter = 10  # (in um)
    Ad = (
        np.pi * (diode_diameter / 2.0 * 1e-6) ** 2
    )  # Surface of the diode, assumed to be circular (in cm)

    # Initial value  of the diode capacitance
    co = Ad * np.sqrt(
        (Q * eps * EPS0)
        / (2 * vbi)
        * ((n_acceptor * 1e6 * n_donor * 1e6) / (n_acceptor * 1e6 + n_donor * 1e6))
    )

    # Resolution of 2nd order equation
    b = -2.0 * co / fixed_capa
    a = -1.0
    c = (
        (1 - vbias / vbi)
        + 2.0 * co / fixed_capa * np.sqrt(1.0 - vbias / vbi)
        - array_2d * Q / (fixed_capa * vbi)
    )
    discriminant = b ** 2 - 4.0 * c * a
    u1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
    v1 = (1.0 - u1 ** 2) * vbi - vbias  # is substracted it only deals with offset level

    array = np.copy(v1)  # unit if V, voltage at the gate of the pixel SFD
    non_linear_signal = array.astype("float64")

    return non_linear_signal


def output_node_linearity_non_linearity_model_2(
    detector: Detector,
) -> None:
    """
    Parameters
    ----------
    detector

    Returns
    -------

    """
    signal_mean_array = detector.pixel.array.astype("float64")
    signal_non_linear = compute_physical_non_linearity_model_2(
        array_2d=signal_mean_array,
        temperature=detector.environment.temperature,
        vbias=-0.220,
        fixed_capa=50.0 * 1e-15,
    )
    detector.signal.array = signal_non_linear


# -----------------------------------------------------------------------
def compute_physical_non_linearity_model_3(
    detector,
    temperature: float,  # Detector operating temperature
    fixed_capa: float,  # Additionnal fixed capacitance
) -> np.ndarray:
    """
    Parameters
    ----------
    detector

    Returns
    -------

    """
    # Derivation of Cd concentration in the alloy,  it depends on cutoff wavelength and targeted operating temperature
    # Here we are considering the case where the detector is operated at its nominal temperature, it might not be always the case
    cutoff = 2.1  # Can be extracted from CMOS characteristics ?
    Eg_targeted = 1.24 / cutoff  # cutoff is um and Eg in eV
    xcd = np.linspace(0.2, 0.6, 1000)
    targeted_operating_temperature = temperature
    Eg_calculated = hgcdte_bandgap(
        xcd, targeted_operating_temperature
    )  # Expected bandgap
    index = np.where(Eg_calculated > Eg_targeted)[0][0]
    x_cd = xcd[index]  # Targeted cadmium concentration in the HgCdTe alloy
    # Calculate the effective bandgap value at the temperature at which simulations are performed.
    Eg = hgcdte_bandgap(x_cd, temperature)

    # Acceptor and donor doping concentrations
    n_acceptor = 1e18  # in atom/cm3
    n_donor = 3e15  # in atoms/cm3

    # Surface of the diode, assumed to be planar
    phi_implant = 6  # in um
    d_implant = 1  # in um

    sat_current = 0.002
    ideality_factor = 1.34
    v_reset = 0.0  # in V
    d_sub = 0.220  # in V
    if detector.signal.array[5, 5] == 0:
        detector.signal.array = v_reset * np.ones(
            (detector.geometry.row, detector.geometry.col)
        )

    # detector.signal.array should be expressed in unit of mV. It is the bias at the gate of the pixel SFD
    det_polar = euler(
        detector.time - detector.time_step,
        detector.time,
        200,
        vbias=np.ravel(detector.signal.array) - d_sub,
        phi_implant=phi_implant,
        d_implant=d_implant,
        n_acceptor=n_acceptor,
        n_donor=n_donor,
        x_cd=x_cd,
        temperature=temperature,
        photonic_current=np.ravel(detector.photon.array) / detector.time_step,
        fixed_capa=fixed_capa,
        sat_current=sat_current,
        n=ideality_factor,
    )

    array = np.reshape(
        det_polar + d_sub, (detector.geometry.row, detector.geometry.col)
    )
    non_linear_signal = array.astype("float64")

    return non_linear_signal


def output_node_linearity_non_linearity_model_3(
    detector: Detector,
) -> None:
    """
    Parameters
    ----------
    detector

    Returns
    -------

    """
    signal_non_linear = compute_physical_non_linearity_model_3(
        detector, temperature=detector.environment.temperature, fixed_capa=50.0 * 1e-15
    )
    detector.signal.array = signal_non_linear


# -------------------------------------------------------------------------
def hgcdte_bandgap(x_cd, temperature: float):
    """
    This expression of the Gap of HgCdTe is valid for a Cadmium concentration between 0.2 and 0.6.
    Over a wide range of temperature beteween 4K and 300K

    Ref : Hansen, G. L., Schmit, J. L., & Casselman, T. N. (1982).
          Energy gap versus alloy composition and temperature in Hg1− x Cd x Te.
          Journal of Applied Physics, 53(10), 7099-7101.

    INPUT : x_cd (float) cadmium composition between 0 and 1
            T : (float) Temperature in K
    OUTPUT : Bandgap energy in eV
    """
    if isinstance(x_cd, float) and not (0.2 <= x_cd <= 0.6):
        print(
            "WARNING: Hansen bangap expression used out of its nomimal application range. \
                x_cd must be between 0.2 and 0.6"
        )

    if not (4 <= temperature <= 300):
        print(
            "WARNING: Hansen bangap expression used out of its nomimal application range. \
                temperature must be between 4K and 300K"
        )

    return (
        -0.302
        + 1.93 * x_cd
        - 0.81 * x_cd ** 2
        + 0.832 * x_cd ** 3
        + 5.35 * 1e-4 * (1 - 2 * x_cd) * temperature
    )

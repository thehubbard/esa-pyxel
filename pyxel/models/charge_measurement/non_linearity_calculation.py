#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:39:18 2021

@author: tpichon
"""
import numpy as np
from numba import njit
from numba.experimental import jitclass
from numba import float64

spec = [
    ('m_electron', float64),
    ('k_b', float64),
    ('electron_charge', float64),
    ('eps_0', float64),
]


#@jitclass(spec)
class Constants(object):
    def __init__(self):
        self.m_electron = 9.10938356e-31  # kg
        self.k_b = 1.38064852e-23  #
        self.electron_charge = 1.60217662e-19
        self.eps_0 = 8.85418782e-12


# --------------------------------------------------------------
#@njit
def build_in_potential(
        temperature: float,
        n_acceptor: float,
        n_donor: float,
        # x_cd: float,
        n_intrinsic: float,
):
    """

    Parameters
    ----------
    Vbias : TYPE, optional
        DESCRIPTION. The default is None.
    R_implant : TYPE, optional
        DESCRIPTION. The default is None.
    epsilon : TYPE, optional
        DESCRIPTION. The default is None.
    Nacceptor : TYPE, optional
        DESCRIPTION. The default is None.
    Ndonor : TYPE, optional
        DESCRIPTION. The default is None.
    xCd : TYPE, optional
        DESCRIPTION. The default is None.
    T : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """

    # Calcul des grandeurs semiconducteurs
    # eg = hgcdte_bandgap(x_cd=x_cd, temperature=temperature)  # Ok for HgCdTe
    # ni = setni(T, Eg=eg, Nc=nc, Nv=nv) if Ni is None else Ni

    # Definition of the constant
    # kb = const.k
    # q = const.e

    const = Constants()

    return (
            const.k_b
            * temperature
            / const.electron_charge
            * np.log(n_acceptor * n_donor / n_intrinsic ** 2)
    )


# -------------------------------------------------------------------------------
#@njit
def w_dep(
        v_bias: np.ndarray,
        epsilon: float,
        n_acceptor: float,
        n_donor: float,
        x_cd: float,
        temperature: float,
):
    """

    Parameters
    ----------
    v_bias : TYPE, optional
        DESCRIPTION. The default is None.
    r_implant : TYPE, optional
        DESCRIPTION. The default is None.
    epsilon : TYPE, optional
        DESCRIPTION. The default is None.
    n_acceptor : TYPE, optional
        DESCRIPTION. The default is None.
    n_donor : TYPE, optional
        DESCRIPTION. The default is None.
    x_cd : TYPE, optional
        DESCRIPTION. The default is None.
    temperature : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    # definition of variable
    # V = -0.50 if vbias is None else vbias  # Bias applied to the junction
    # Na = 1e17 if n_acceptor is None else n_acceptor  # Acceptor concentratuion
    # Nd = 1e14 if n_donor is None else n_donor  # Donor concentration
    # xcd = 0.515 if x_cd is None else x_cd  # Cd concentration
    # temp = 80 if temperature is None else temperature  # Temperature in K
    # eps = (
    #     20.5 - 15.6 * xcd + 5.7 * xcd ** 2 if epsilon is None else epsilon
    # )  # Static dielectric constant

    const = Constants()

    # Calculation of build in potenial
    v_bi = build_in_potential(
        temperature=temperature,
        n_acceptor=n_acceptor,
        n_donor=n_donor,
        n_intrinsic=ni_hansen(temperature, x_cd),
    )

    return np.sqrt(
        2.0
        * epsilon
        * const.eps_0
        / const.electron_charge
        * (n_acceptor * 1e6 + n_donor * 1e6)
        / (n_acceptor * 1e6 * n_donor * 1e6)
        * (v_bi - v_bias)
    )


# -------------------------------------------------------------------------
#@njit
def hgcdte_bandgap(x_cd: float, temperature: float):
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
    # if isinstance(x_cd, float) and not (0.2 <= x_cd <= 0.6):
    #     print(
    #         "WARNING: Hansen bangap expression used out of its nomimal application range. \
    #             x_cd must be between 0.2 and 0.6"
    #     )
    #
    # if not (4 <= temperature <= 300):
    #     print(
    #         "WARNING: Hansen bangap expression used out of its nomimal application range. \
    #             temperature must be between 4K and 300K"
    #     )

    return (
            -0.302
            + 1.93 * x_cd
            - 0.81 * x_cd ** 2
            + 0.832 * x_cd ** 3
            + 5.35 * 1e-4 * (1 - 2 * x_cd) * temperature
    )


# -------------------------------------------------------------------------------
#@njit
def ni_hansen(temperature, x_cd):
    """
    ni : intrinsic carrier concentration for HgCdTe

    INPUT : T Temperature en K, float
            x Cd concentration
    OUTPUT : ni

    Ref : G.L. Hansen and J.L. Schmit  J. Applied Physics, 54, 1639 (1983)
    """

    const = Constants()

    Eg = hgcdte_bandgap(x_cd, temperature)
    return (
            (
                    5.585
                    - 3.820 * x_cd
                    + 1.753 * 1e-3 * temperature
                    - 1.364 * 1e-3 * temperature * x_cd
            )
            * 1e14
            * Eg ** 0.75
            * temperature ** 1.5
            * np.exp(-const.electron_charge * Eg / (2 * const.k_b * temperature))
    )


# ==============================================================
#                   CYLINDRICAL PN JUNCTION
# ==============================================================
#@njit
def capa_pn_junction_cylindrical(
        v_bias: np.ndarray,
        phi_implant: float,
        d_implant: float,
        epsilon: float,
        n_acceptor: float,
        n_donor: float,
        x_cd: float,
        temperature: float,
):
    """ """
    # Definition of the variable
    # vb = -0.50 if vbias is None else vbias  # Bias applied to the junction
    # phi_imp = (
    #     6 * 1e-6 if phi_implant is None else phi_implant * 1e-6
    # )  # size of the implantation diameter
    # d_imp = (
    #     1 * 1e-6 if d_implant is None else d_implant * 1e-6
    # )  # Depth of the implantation
    # Na = 1e17 if n_acceptor is None else n_acceptor  # Acceptor concentration in cm3
    # Nd = 1e14 if n_donor is None else n_donor  # Donor concentration in cm3
    # xcd = 0.515 if x_cd is None else x_cd  # Cd concentration
    # T = 80 if temperature is None else temperature  # Temperature in K

    # Definition of the constants
    const = Constants()
    eps0 = const.eps_0
    # Calculation of w
    w = w_dep(
        v_bias=v_bias,
        epsilon=epsilon,
        n_acceptor=n_acceptor,
        n_donor=n_donor,
        x_cd=x_cd,
        temperature=temperature,
    )

    ao = np.pi * epsilon * eps0 * ((phi_implant / 2.0) ** 2 + d_implant * phi_implant)
    bo = np.pi * epsilon * eps0 * 2.0 * (d_implant + phi_implant)
    co = 3.0 * np.pi * epsilon * eps0
    return ao * 1.0 / w + bo + co * w


# ----------------------------------------------------------------
# CYLINDRICAL PN JUNCTION
#@njit
def dv_dt_cylindrical(
        v_bias: np.ndarray,
        photonic_current: np.ndarray,
        fixed_capacitance: float,
        sat_current: float,
        n: float,
        phi_implant: float,
        d_implant: float,
        n_acceptor: float,
        n_donor: float,
        x_cd: float,
        temperature: float,
):
    """
    Use this diode discharge equation when considering a cylindrical diode

    Parameters
    ----------
    v_bias : FLOAT, Diode polarisation
    photonic_current : FLOAT, Photonic current
    fixed_capacitance : FLOAT, Fixed capacitance
    sat_current : FLOAT, Saturating current under dark conditions
    n : FLOAT, Quality factor
    phi_implant : FLOAT, diameter of the implantation
    d_implant : FLOAT, depth of the implantation
    n_acceptor : FLOAT, density of acceptors
    n_donor : FLOAT, density of donor
    x_cd : FLOAT, cadmium concentrations
    temperature : FLOAT, temperature of the sample

    Returns
    -------
    dv_dt : FLOAT Polarisation evolution per second.
    """
    const = Constants()
    # Constant definition
    e, k = const.electron_charge, const.k_b
    # Capacitance calculation as a function of Bias

    epsilon = (
        20.5 - 15.6 * x_cd + 5.7 * x_cd**2
    )  # Static dielectric constant, this value is ok for HgCdTe

    c = capa_pn_junction_cylindrical(
        v_bias=v_bias,
        phi_implant=phi_implant,
        epsilon=epsilon,
        d_implant=d_implant,
        n_acceptor=n_acceptor,
        n_donor=n_donor,
        x_cd=x_cd,
        temperature=temperature,
    )

    return -(
            e * sat_current * (np.exp(e * v_bias / (n * k * temperature)) - 1)
            - e * photonic_current
    ) / (fixed_capacitance + c)


# ----------------------------------------------------------------
# Euler method
#@njit
def euler(
        time_step: float,
        nb_pts: int,
        v_bias: np.ndarray,
        phi_implant: float,
        d_implant: float,
        n_acceptor: float,
        n_donor: float,
        x_cd: float,
        temperature: float,
        photonic_current: np.ndarray,
        fixed_capacitance: float,
        sat_current: float,
        n: float,
):
    """
    Use this diode discharge equation when considering a cylindrical diode

    Parameters
    ----------
    time_step
    nb_pts : FLOAT number of points
    v_bias : FLOAT, Diode polarisation
    phi_implant : FLOAT, diameter of the implantation
    d_implant : FLOAT, depth of the implantation
    n_acceptor : FLOAT, density of acceptors
    n_donor : FLOAT, density of donor
    x_cd : FLOAT, cadmium concentrations
    temperature : FLOAT, temperature of the sample
    photonic_current : FLOAT, Photonic current
    fixed_capacitance : FLOAT, Fixed capacitance
    sat_current : FLOAT, Saturating current under dark conditions
    n : FLOAT, Quality factor

    Returns
    -------
    V : FLOAT: voltage at the gate of the pixel SFD
    """
    # Calculating step size
    h = time_step / nb_pts
    # Initialization of variables
    v = [v_bias]
    # ==============================
    #           SOLVE THE DIFFERENTIAL EQUATION
    # ==============================
    # EULER Method to solve the differential equation
    for i in range(1, nb_pts):
        slope = dv_dt_cylindrical(
            v[-1],
            photonic_current,
            fixed_capacitance,
            sat_current,
            n,
            phi_implant,
            d_implant,
            n_acceptor,
            n_donor,
            x_cd,
            temperature,
        )
        # Calculate new bias
        yn = v[-1] + h * slope
        # Saved the old bias
        v.append(yn)

    # Return bias evolution, capacitance and time
    return v[-1]

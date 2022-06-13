#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:39:18 2021

@author: tpichon
"""
import numpy as np
import scipy.constants as const


# --------------------------------------------------------------
def buildin_potential(T, Nacceptor, Ndonor, xcd, Ni=None):
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
    eg = hgcdte_bandgap(xcd, T)  # Ok for HgCdTe
    ni = setni(T, Eg=eg, Nc=nc, Nv=nv) if Ni is None else Ni

    # Definition of the constant
    kb = const.k
    q = const.e

    return kb * T / q * np.log(Nacceptor * Ndonor / ni ** 2)


# -------------------------------------------------------------------------------
def wdep(
    vbias=None, epsilon=None, n_acceptor=None, n_donor=None, x_cd=None, temperature=None
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
    V = -0.50 if vbias is None else vbias  # Bias applied to the junction
    Na = 1e17 if n_acceptor is None else n_acceptor  # Acceptor concentratuion
    Nd = 1e14 if n_donor is None else n_donor  # Donor concentration
    xcd = 0.515 if x_cd is None else x_cd  # Cd concentration
    temp = 80 if temperature is None else temperature  # Temperature in K
    eps = (
        20.5 - 15.6 * xcd + 5.7 * xcd ** 2 if epsilon is None else epsilon
    )  # Static dielectric constant

    # Definition of constant
    e = const.e
    eps0 = const.epsilon_0
    # Calculation of build in potenial
    Vbi = buildin_potential(temp, Na, Nd, xcd, Ni=ni_hansen(temp, xcd))

    return np.sqrt(
        2.0 * eps * eps0 / e * (Na * 1e6 + Nd * 1e6) / (Na * 1e6 * Nd * 1e6) * (Vbi - V)
    )


# -------------------------------------------------------------------------
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


# -------------------------------------------------------------------------------
def ni_hansen(T, xCd):
    """
    ni : intrinsic carrier concentration for HgCdTe

    INPUT : T Temperature en K, float
            x Cd concentration
    OUTPUT : ni

    Ref : G.L. Hansen and J.L. Schmit  J. Applied Physics, 54, 1639 (1983)
    """
    k = const.k
    e = const.e

    Eg = hgcdte_bandgap(xCd, T)
    return (
        (5.585 - 3.820 * xCd + 1.753 * 1e-3 * T - 1.364 * 1e-3 * T * xCd)
        * 1e14
        * Eg ** 0.75
        * T ** 1.5
        * np.exp(-e * Eg / (2 * k * T))
    )


# ==============================================================
#                   CYLINDRICAL PN JUNCTION
# ==============================================================
def capa_pn_junction_cylindrical(
    vbias=None,
    phi_implant=None,
    d_implant=None,
    epsilon=None,
    n_acceptor=None,
    n_donor=None,
    x_cd=None,
    temperature=None,
):
    """ """
    # Definition of the variable
    vb = -0.50 if vbias is None else vbias  # Bias applied to the junction
    phi_imp = (
        6 * 1e-6 if phi_implant is None else phi_implant * 1e-6
    )  # size of the implantation diameter
    d_imp = (
        1 * 1e-6 if d_implant is None else d_implant * 1e-6
    )  # Depth of the implantation
    Na = 1e17 if n_acceptor is None else n_acceptor  # Acceptor concentration in cm3
    Nd = 1e14 if n_donor is None else n_donor  # Donor concentration in cm3
    xcd = 0.515 if x_cd is None else x_cd  # Cd concentration
    T = 80 if temperature is None else temperature  # Temperature in K
    eps = (
        20.5 - 15.6 * xcd + 5.7 * xcd ** 2 if epsilon is None else epsilon
    )  # Static dielectric constant, this value is ok for HgCdTe

    # Definition of the constants
    eps0 = const.epsilon_0
    # Calculation of W
    W = wdep(vbias=vb, epsilon=eps, n_acceptor=Na, n_donor=Nd, x_cd=xcd, temperature=T)

    ao = np.pi * eps * eps0 * ((phi_imp / 2.0) ** 2 + d_imp * phi_imp)
    bo = np.pi * eps * eps0 * 2.0 * (d_imp + phi_imp)
    co = 3.0 * np.pi * eps * eps0
    return ao * 1.0 / W + bo + co * W


# ----------------------------------------------------------------
# CYLINDRICAL PN JUNCTION
def dv_dt_cylindrical(
    vbias,
    photonic_current,
    fixed_capa,
    sat_current,
    n,
    phi_implant,
    d_implant,
    n_acceptor,
    n_donor,
    x_cd,
    temperature,
):
    """
    Use this diode discharge equation when considering a cylindrical diode

    Parameters
    ----------
    vbias : FLOAT, Diode polarisation
    photonic_current : FLOAT, Photonic current
    fixed_capa : FLOAT, Fixed capacitance
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
    # Constant definition
    e, k = const.e, const.k
    # Capacitance calculation as a function of Bias
    C = capa_pn_junction_cylindrical(
        vbias=vbias,
        phi_implant=phi_implant,
        d_implant=d_implant,
        n_acceptor=n_acceptor,
        n_donor=n_donor,
        x_cd=x_cd,
        temperature=temperature,
    )
    return -(
        e * sat_current * (np.exp(e * vbias / (n * k * temperature)) - 1)
        - e * photonic_current
    ) / (fixed_capa + C)


# ----------------------------------------------------------------
# Euler method
def euler(
    t0,
    t_end,
    nb_pts,
    vbias=None,
    phi_implant=None,
    d_implant=None,
    n_acceptor=None,
    n_donor=None,
    x_cd=None,
    temperature=None,
    photonic_current=None,
    fixed_capa=None,
    sat_current=None,
    n=None,
):
    """
    Use this diode discharge equation when considering a cylindrical diode

    Parameters
    ----------
    t0 : FLOAT, init time
    t_end : FLOAT, end time
    nb_pts : FLOAT number of points
    vbias : FLOAT, Diode polarisation
    phi_implant : FLOAT, diameter of the implantation
    d_implant : FLOAT, depth of the implantation
    n_acceptor : FLOAT, density of acceptors
    n_donor : FLOAT, density of donor
    x_cd : FLOAT, cadmium concentrations
    temperature : FLOAT, temperature of the sample
    photonic_current : FLOAT, Photonic current
    fixed_capa : FLOAT, Fixed capacitance
    sat_current : FLOAT, Saturating current under dark conditions
    n : FLOAT, Quality factor

    Returns
    -------
    V : FLOAT: voltage at the gate of the pixel SFD
    """
    # Calculating step size
    h = (t_end - t0) / nb_pts
    # Initialization of variables
    V = [vbias]
    # ==============================
    #           SOLVE THE DIFFERENTIAL EQUATION
    # ==============================
    # EULER Method to solve the differential equation
    for i in range(1, nb_pts):
        slope = dv_dt_cylindrical(
            V[-1],
            photonic_current,
            fixed_capa,
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
        yn = V[-1] + h * slope
        # Saved the old bias
        V.append(yn)

    # Return bias evolution, capacitance and time
    return V[-1]

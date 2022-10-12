#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
"""Utility functions for mercury cadmium telluride alloy material properties."""
import numpy as np


class Mercadtel():
    """Useful functions to compute properties of mercury cadmium telluride alloy."""


def lambda_e(lambda_cutoff: float) -> float:
    """Compute lambda_e.

    Parameters
    ----------
    lambda_cutoff: int
        Cut-off wavelength of the detector.

    Returns
    -------
    le: float
    """
    lambda_scale = 0.200847413  # um
    lambda_threshold = 4.635136423  # um
    pwr = 0.544071282
    if lambda_cutoff < lambda_threshold:
        le = lambda_cutoff / (
            1
            - ((lambda_scale / lambda_cutoff) - (lambda_scale / lambda_threshold))
            ** pwr
        )
    else:
        le = lambda_cutoff
    return le


def energy_bandgap(cut_off):
    """
    Computes the energy of the MCT bandgap as a function of its cut-off wavelength
    :param cut_off: the cut_off wavelength in micrometer
    :return: the bandgap energy in eV
    """
    e_c = 1.24 / lambda_e(cut_off)
    return e_c


def eg_hansen_inverse(eg, temperature):
    """Hansen formulation inversion
    Returns the cadmium fraction as a function of the temperature and the bandgap energy
    Ref Biblio: G. L. HANSEN, J. L. SCHMIT, T. N. CASSELMAN, J. Appl. Phys., 53 (1982), p. 7099-7101
    Eg: gap energy [eV]
    temperature:  Temperature [K]
    returns xx: MCT Cd concentration"""
    aa = 0.302
    bb = 1.93
    cc = 0.535/1000
    dd = 0.810
    ee = 0.832
    xx = ((np.sqrt(-32. * cc ** 3. * ee * temperature ** 3 - (-27. * cc ** 2. * ee ** 2 + (36. * cc ** 2. * dd - 48. * bb * cc ** 2) * ee + 4. * cc ** 2. * dd ** 2) * temperature ** 2 - ((54. * cc * ee ** 2 - 36. * cc * dd * ee) * eg + 54. * aa * cc * ee ** 2 + ((-18. * bb - 36. * aa) * cc * dd + 24. * bb ** 2. * cc) * ee + 4. * cc * dd ** 3 - 4. * bb * cc * dd ** 2) * temperature + 27. * ee ** 2. * eg ** 2 - (-54. * aa * ee ** 2 + 18. * bb * dd * ee - 4. * dd ** 3) * eg + 27. * aa ** 2. * ee ** 2 - (18. * aa * bb * dd - 4. * bb ** 3) * ee + 4. * aa * dd ** 3 - bb ** 2. * dd ** 2)) / (6. * np.sqrt(3) * ee ** 2) + (cc * (18. * dd * ee * temperature - 27. * ee ** 2. * temperature) + 27. * ee ** 2. * eg + 27. * aa * ee ** 2 - 9. * bb * dd * ee + 2. * dd ** 3) / (54. * ee ** 3)) ** (1. / 3) - (-6. * cc * ee * temperature + 3. * bb * ee - dd ** 2) / (9. * ee ** 2. * ((np.sqrt(-32. * cc ** 3. * ee * temperature ** 3 - (-27. * cc ** 2. * ee ** 2 + (36. * cc ** 2. * dd - 48. * bb * cc ** 2) * ee + 4. * cc ** 2. * dd ** 2) * temperature ** 2 - ((54. * cc * ee ** 2 - 36. * cc * dd * ee) * eg + 54. * aa * cc * ee ** 2 + ((-18. * bb - 36. * aa) * cc * dd + 24. * bb ** 2. * cc) * ee + 4. * cc * dd ** 3 - 4. * bb * cc * dd ** 2) * temperature + 27. * ee ** 2. * eg ** 2 - (-54. * aa * ee ** 2 + 18. * bb * dd * ee - 4. * dd ** 3) * eg + 27. * aa ** 2. * ee ** 2 - (18. * aa * bb * dd - 4. * bb ** 3) * ee + 4. * aa * dd ** 3 - bb ** 2. * dd ** 2)) / (6. * np.sqrt(3) * ee ** 2) + (cc * (18. * dd * ee * temperature - 27. * ee ** 2. * temperature) + 27. * ee ** 2. * eg + 27. * aa * ee ** 2 - 9. * bb * dd * ee + 2. * dd ** 3) / (54. * ee ** 3)) ** (1. / 3)) + dd / (3. * ee)
    return xx


def eg_hansen(x, temperature):
    """
    Computes the MCT Energy gap computation as a function of the Cd fraction
    Ref Biblio: G. L. HANSEN, J. L. SCHMIT, T. N. CASSELMAN, J. Appl. Phys., 53 (1982), p. 7099-7101
    x: MCT Cd concentration
    temperature: temperature [K]
    Returns eg: gap energy [eV]
    """
    eg = -0.302+1.93*x-0.810*x**2+0.832*x**3+0.535*(1-2*x)*temperature/1000
    return eg


def absorption_coefficient(e, x, temperature):
    """
    computation of the absorption coefficient of MCT alloy using a non parabolic band model and an urbach tail
    Ref: Moazzami, JEM 34(6) p773 (2005)
    :param e: photon energy [eV]
    :param x: MCT Cd composition fraction
    :param temperature: temperature [K]
    :return: the absorbtion coefficient
    """
    w = 4.96e-3
    K = -20060 + 115750 * x + 32.43 * temperature - 64170 * x ** 2 + 0.43231 * temperature ** 2 - 101.92 * x * temperature
    n = 0.74487 - 0.44513 * x + 1e-4 * (7.99 - 7.57 * x) * temperature
    Eg = eg_hansen(x, temperature)
    delta = (Eg + w * (n - 1)) ** 2 + 4 * w * Eg
    Eo = (Eg + w * (n - 1) + np.sqrt(delta)) / 2
    alphao = K * (Eo - Eg) ** n / Eo
    alpha1 = alphao * np.exp((e - Eo) / w)
    alpha2 = K * (e - Eg) ** n / e
    if e < Eo:
        alpha = alpha1
    else:
        alpha = alpha2
    return alpha


def density(x):
    """
    Compute the MCT density as a function of the cadmium fraction
    :param x: the cadmium fraction
    :return: the density in kg/m3
    """
    me_density = 5.43
    cd_density = 8.65
    te_density = 6.24
    mct_density = 0.5*(x/cd_density + (1-x)/me_density) + 0.5/te_density
    mct_density = 1./mct_density
    return mct_density

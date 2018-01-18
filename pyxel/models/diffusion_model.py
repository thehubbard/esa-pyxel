#   --------------------------------------------------------------------------
#   Copyright 2017 SCI-FIV, ESA (European Space Agency)
#   --------------------------------------------------------------------------
""" PyXel! Charge diffusion and collection model extracted from TARS
"""

import copy
from math import sqrt, log

import numpy as np
from astropy import units as u
from astropy.units import cds
from scipy.special import erf

from pyxel.detectors.ccd import CCD

cds.enable()


def diffusion(ccd: CCD) -> CCD:
    """ TBD """

    new_ccd = copy.deepcopy(ccd)

    diff = Diffusion(new_ccd)

    collected_charge_list = []  # type: list

    cluster_generator = [cluster for cluster in new_ccd.charge_list]

    for cluster in cluster_generator:

        # Modifying the positions and shapes of charge clusters in list of Charge class instances

        # sigma_hiraga = diff.hiraga_diffusion_model(cluster)

        # sigma_janesick = diff.janesick_diffusion_model(cluster)

        sigma = 1.0     # temporarily
        collected_charge = diff.gaussian_pixel_separation(cluster, sigma, sigma)

        collected_charge_list += collected_charge

    # Overwrite the list of charge clusters in the new_ccd object because Charge attributes have changed
    # new_ccd.charge_list = collected_charge_list
    new_ccd.charge = collected_charge_list      # TEMPORARY

    return new_ccd


class Diffusion:
    """ TBD """

    def __init__(self, ccd):

        self.ccd = ccd

        # Here is an image of all the last simulated CRs events on the CCD
        self.pcmap_last = np.zeros((self.ccd.row, self.ccd.col))

    # DIFFUSION
    def janesick_diffusion_model(self, cluster):
        """ TBD """

        # Initial cloud diameter:
        c_init = 0.0171 * (cluster.energy.value ** 1.75)

        # 10 keV deposited by an X-ray photon resultsParticle a 1 um diameter charge (e-h) cloud
        # CCD Advances For X - Ray Scientific Measurements In 1985,
        # James Janesick et al.
        # deltaE != cluster.number / u.electron * self.ccd.material_ionization_energy / (1000 * u.eV)
        # deltaE == kin. energy of an electron
        # By analogy with high - energy electron beam interaction with silicon, one can approximate the
        # energy / depth relationship as R = k * E**n , where k and n are numerical constants
        # for the material and R is the penetration depth.

        x_backside = self.ccd.total_thickness      # um  # boundary of the ccd backside
        x_ff = self.ccd.field_free_zone            # um # boundary of the field-free region near backside
        x_p = self.ccd.depletion_zone              # um # boundary of the depletion region near ccd channel

        # z position of charge generation event relative to the backside (>= 0)
        x_a = x_backside + cluster.initial_position[2]

        n_acceptor = 1e15 * u.cm ** (-3)

        c_field_free = 0.0
        # c_field = 0.0

        if x_a == 0:
            raise ValueError

        # Cloud diameter after passing through a thin field-free region:
        if 0 < x_a < x_ff:
            c_field_free = 2 * x_ff * (1 - (x_a / x_ff) ** 2) ** 0.5

        # Cloud diameter after passing through the field region:
        # if x_ff <= x_a < x_ff + x_p:
        c_field = (-2 * (5.1e-6 * (1e15 / n_acceptor.value) ** 0.5) ** 2 * log((x_a - x_ff) / x_p)) ** 0.5
        c_field_max = 1.85e-5 * (1e15 / n_acceptor.value) ** 0.5
        c_field = max(c_field, c_field_max)

        # Charges already created inside CCD channel, not needed to diffuse them
        if x_ff + x_p <= x_a:
            c_field = 0.0

        # Final cloud diameter: (um)
        c_diameter = sqrt(c_init ** 2 + c_field_free ** 2 + c_field ** 2) ** 0.5 * u.um

        return c_diameter

    # DIFFUSION
    def hiraga_diffusion_model(self, cluster):
        """
        spread the particle into the material and compute the density and size of the electronic cloud generated
        at each step
        """
        pass
        # eps_rel = 11.8                       # TODO: implement this in CCDDetector class
        # eps_si = eps_rel * cds.eps0

        # n_acceptor = 1e15 * u.cm**(-3)       # TODO: implement this in CCDDetector class

        # voltage = cds.e * n_acceptor / (2 * eps_si) * l_dep**2    # V
        # TODO: implement this in CCDDetector class
        # assumptions made: V=0, dV/dz = 0 at z = ld
        # voltage = 50 * u.V
        # voltage = self.ccd.bias_voltage

        # depletion depth                      # TODO: implement this in CCDDetector class
        # l_dep = sqrt(2 * eps_si * voltage / (cds.e * n_acceptor))   # cm
        # l_dep = self.ccd.depletion_zone

        # critical field
        # efield_crit = 1.01 * u.V/u.cm * self.ccd.temperature ** 1.55        # V/cm
        # For instance, typical value at T = 210 K are vs = 1.46e7 cm*sec−1, Ecrit = 4.4e3 V*cm−1 and α = 0.88
        # Ecrit ~ 1e2 - 1e4 V*cm−1

        # electron velocity saturation parameter
        # sat = u.e * n_acceptor * l_dep / (eps_si * efield_crit)

        # r_final =

        #     spreading across entire depletion region
        # self.cfr = self.con * sqrt(self.sat + bound)

        # return r_final

    # ELECTRON COLLECTION
    def gaussian_pixel_separation(self, cluster, sig_ac, sig_al):
        """
        Compute the charge collection function to determine the number of electron collected by each pixel based on the
        generated electronic cloud shape

        :param cluster:
        :param float sig_ac: diameter of the resulting electronic cloud in the AC (across scan, vertical) dimension
        :param float sig_al: diameter of the resulting electronic cloud in the AL (along scan, horizontal) dimension
        """

        self.pcmap_last[:, :] = 0
        px = []
        py = []

        dx = (cluster.initial_position[0] - self.ccd.pix_ver_size
              * int(cluster.initial_position[0] / self.ccd.pix_ver_size))
        dy = (cluster.initial_position[1] - self.ccd.pix_hor_size
              * int(cluster.initial_position[1] / self.ccd.pix_hor_size))

        try:
            int(4 * sig_ac / self.ccd.pix_ver_size)  # WTF?
        except ValueError:
            print(sig_ac, cluster.number)

        x_steps = int(4 * sig_ac / self.ccd.pix_ver_size)

        if x_steps > 49:  # WHY????
            x_steps = 49
        if x_steps < 1:
            x_steps = 1

        y_steps = int(4 * sig_al / self.ccd.pix_hor_size)
        if y_steps > 49:
            y_steps = 49
        if y_steps < 1:
            y_steps = 1

        for xi in np.arange(-(x_steps * self.ccd.pix_ver_size + dx),
                            ((x_steps + 1) * self.ccd.pix_ver_size - dx),
                            self.ccd.pix_ver_size):

            if sig_ac != 0:
                case1 = (xi + self.ccd.pix_ver_size) / 1.41 / sig_ac
                case2 = xi / 1.41 / sig_ac
            else:
                case1 = 0
                case2 = 0

            px.append((erf(case1) - erf(case2)) / 2)

        for yi in np.arange(-(y_steps * self.ccd.pix_hor_size + dy),
                            ((y_steps + 1) * self.ccd.pix_hor_size - dy),
                            self.ccd.pix_hor_size):

            if sig_al != 0:
                case1 = (yi + self.ccd.pix_hor_size) / 1.41 / sig_al
                case2 = yi / 1.41 / sig_al
            else:
                case1 = 0
                case2 = 0

            py.append((erf(case1) - erf(case2)) / 2)

        cx = 0

        for ix in range(int(cluster.initial_position[0] / self.ccd.pix_ver_size) - x_steps,
                        int(cluster.initial_position[0] / self.ccd.pix_ver_size) + x_steps + 1, 1):

            cy = 0

            for iy in range(int(cluster.initial_position[1] / self.ccd.pix_hor_size) - y_steps,
                            int(cluster.initial_position[1] / self.ccd.pix_hor_size) + y_steps + 1, 1):

                if 0 <= ix < self.ccd.row and 0 <= iy < self.ccd.col:
                    self.pcmap_last[ix, iy] += px[cx] * py[cy] * cluster.number

                cy += 1

            cx += 1

        # diff.pcmap_last = np.rint(diff.pcmap_last).astype(int)
        # diff.total_charge_array += diff.pcmap_last

        return self.pcmap_last

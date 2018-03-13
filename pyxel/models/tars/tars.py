#   --------------------------------------------------------------------------
#   Copyright 2018 SCI-FIV, ESA (European Space Agency)
#   --------------------------------------------------------------------------
"""PyXel! TARS model for charge generation by ionization."""

import logging
import math

import numpy as np
from tqdm import tqdm
import typing as t   # noqa: F401

from pyxel.detectors.detector import Detector
from pyxel.models.tars.simulation import Simulation
from pyxel.models.tars.util import read_data, interpolate_data
from pyxel.pipelines.model_registry import registry

# from astropy import units as u


@registry.decorator('charge_generation', name='tars')
def run_tars(detector: Detector,
             particle_type: str = None,
             initial_energy: t.Union[str, float] = None,
             particle_number: int = None,
             incident_angles: tuple = None,
             starting_position: tuple = None,
             stepping_length: float = None,
             spectrum_file: str = None) -> Detector:
    """TBW.

    :param detector:
    :param particle_type:
    :param initial_energy:
    :param particle_number:
    :param incident_angles:
    :param starting_position:
    :param stepping_length:
    :param spectrum_file:
    :return:
    """
    new_detector = detector

    cosmics = TARS(new_detector)

    if particle_type is None:
        raise ValueError('TARS: Particle type is not defined')
    if particle_number is None:
        raise ValueError('TARS: Particle number is not defined')
    if spectrum_file is None:
        raise ValueError('TARS: Spectrum is not defined')

    if initial_energy is None:
        initial_energy = 'random'       # TODO
    if incident_angles is None:
        incident_angles = ('random', 'random')
    if starting_position is None:
        starting_position = ('random', 'random', 0.)
        # starting_position = ('random', 'random', 'random') -> snowflakes (radioactive decay inside detector)

    if stepping_length is None:
        stepping_length = 1.    # um

    cosmics.set_particle_type(particle_type)                # MeV
    cosmics.set_initial_energy(initial_energy)              # MeV
    cosmics.set_particle_number(particle_number)            # -
    cosmics.set_incident_angles(incident_angles)            # rad
    cosmics.set_starting_position(starting_position)        # um
    cosmics.set_stepping_length(stepping_length)            # um
    cosmics.set_particle_spectrum(spectrum_file)

    cosmics.run()

    return new_detector


class TARS:
    """TBW."""

    def __init__(self, detector: Detector) -> None:
        """TBW.

        :param detector:
        """
        self.detector = detector

        self.part_type = None
        self.init_energy = None
        self.particle_number = None
        self.angle_alpha = None
        self.angle_beta = None
        self.position_ver = None
        self.position_hor = None
        self.position_z = None
        self.step_length = None

        self.sim_obj = Simulation(self.detector)
        self.charge_obj = self.detector.charges
        self.log = logging.getLogger(__name__)

    def set_particle_type(self, particle_type):
        """TBW.

        :param particle_type:
        :return:
        """
        self.part_type = particle_type

    def set_initial_energy(self, energy):
        """TBW.

        :param energy:
        :return:
        """
        self.init_energy = energy

    def set_particle_number(self, number):
        """TBW.

        :param number:
        :return:
        """
        self.particle_number = number

    def set_incident_angles(self, angles):
        """TBW.

        :param angles:
        :return:
        """
        alpha, beta = angles
        self.angle_alpha = alpha
        self.angle_beta = beta

    def set_starting_position(self, start_position):
        """TBW.

        :param start_position:
        :return:
        """
        position_vertical, position_horizontal, position_z = start_position
        self.position_ver = position_vertical
        self.position_hor = position_horizontal
        self.position_z = position_z

    def set_stepping_length(self, stepping):
        """TBW.

        :param stepping:
        :return:
        """
        self.step_length = stepping  # um

    def run(self):
        """TBW.

        :return:
        """
        print("TARS - simulation processing...\n")

        self.sim_obj.parameters(self.part_type,
                                self.init_energy,
                                self.position_ver, self.position_hor, self.position_z,
                                self.angle_alpha, self.angle_beta,
                                self.step_length)

        for _ in tqdm(range(0, self.particle_number)):
            self.sim_obj.event_generation()

        size = len(self.sim_obj.e_num_lst)
        self.sim_obj.e_vel0_lst = [0.] * size
        self.sim_obj.e_vel1_lst = [0.] * size
        self.sim_obj.e_vel2_lst = [0.] * size

        self.charge_obj.add_charge('e',
                                   self.sim_obj.e_num_lst,
                                   self.sim_obj.e_energy_lst,
                                   self.sim_obj.e_pos0_lst,
                                   self.sim_obj.e_pos1_lst,
                                   self.sim_obj.e_pos2_lst,
                                   self.sim_obj.e_vel0_lst,
                                   self.sim_obj.e_vel1_lst,
                                   self.sim_obj.e_vel2_lst)

        # np.save('orig2_edep_per_step_10k', self.sim_obj.edep_per_step)
        # np.save('orig2_edep_per_particle_10k', self.sim_obj.total_edep_per_particle)

        # self.plot_edep_per_step()
        # self.plot_edep_per_particle()
        # self.plot_charges_3d()
        # plt.show()

    # def plot_edep_per_step(self):
    #     plt.figure()
    #     n, bins, patches = plt.hist(self.sim_obj.edep_per_step, 300, facecolor='b')
    #     plt.xlabel('E_dep (keV)')
    #     plt.ylabel('Counts')
    #     plt.title('Histogram of E deposited per step')
    #     # plt.axis([0, 0.003, 0, 1.05*max(n)])
    #     plt.grid(True)
    #     plt.draw()
    #     return n, bins, patches
    #
    # def plot_edep_per_particle(self):
    #     plt.figure()
    #     n, bins, patches = plt.hist(self.sim_obj.total_edep_per_particle, 200, facecolor='g')
    #     plt.xlabel('E_dep (keV)')
    #     plt.ylabel('Counts')
    #     plt.title('Histogram of total E deposited per particle')
    #     # plt.axis([0, 0.4, 0, 1.05*max(n)])
    #     plt.grid(True)
    #     plt.draw()
    #     return n, bins, patches

    # def set_stopping_power(self, stopping_file):
    #     self.sim_obj.stopping_power_function = read_data(stopping_file)
    #     self.sim_obj.energy_max_limit = self.sim_obj.stopping_power_function[-1, 0]

    def set_particle_spectrum(self, file_name):
        """Set up the particle specs according to a spectrum.

        :param string file_name: path of the file containing the spectrum
        """
        spectrum = read_data(file_name)  # nuc/m2*s*sr*MeV

        detector_area = self.detector.geometry.vert_dimension * self.detector.geometry.horz_dimension * 1.0e-8  # cm2

        spectrum[:, 1] *= 4 * math.pi * 1.0e-4 * detector_area  # nuc/s*MeV

        spectrum_function = interpolate_data(spectrum)

        lin_energy_range = np.arange(np.min(spectrum[:, 0]), np.max(spectrum[:, 0]), 0.01)
        flux_dist = spectrum_function(lin_energy_range)

        cum_sum = np.cumsum(flux_dist)
        cum_sum /= np.max(cum_sum)
        self.sim_obj.spectrum_cdf = np.stack((lin_energy_range, cum_sum), axis=1)

        # plt.figure()
        # plt.loglog(lin_energy_range, flux_dist)
        # plt.draw()

        # plt.figure()
        # plt.semilogx(lin_energy_range, cum_sum)
        # plt.draw()

        # plt.show()

    # def plot_charges_3d(self):
    #
    #     # set up a figure twice as wide as it is tall
    #     fig = plt.figure(figsize=plt.figaspect(0.5))
    #     ax = fig.add_subplot(1, 2, 1, projection='3d')
    #
    #     # generator expression
    #     # sum(c.A for c in c_list)
    #     # asc = self.sim_obj.all_charge_clusters[0].position
    #
    #     ############################### need to be fixed
    #     # init_pos0 = [cluster.initial_position[0] for cluster in self.sim_obj.all_charge_clusters]
    #     # init_pos1 = [cluster.initial_position[1] for cluster in self.sim_obj.all_charge_clusters]
    #     # init_pos2 = [cluster.initial_position[2] for cluster in self.sim_obj.all_charge_clusters]
    #     # cluster_size = [cluster.number.value for cluster in self.sim_obj.all_charge_clusters]
    #     #
    #     # ax.scatter(init_pos0, init_pos1, init_pos2, c='b', marker='.', s=cluster_size)
    #     #
    #     # ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    #     #
    #     # pos0 = [cluster.position[0] for cluster in self.sim_obj.all_charge_clusters]
    #     # pos1 = [cluster.position[1] for cluster in self.sim_obj.all_charge_clusters]
    #     # pos2 = [cluster.position[2] for cluster in self.sim_obj.all_charge_clusters]
    #     #
    #     # ax2.scatter(pos0, pos1, 0, c=r', marker='.', s=cluster_size)
    #     #
    #     # ax.set_xlim(0, self.ccd.vert_dimension)
    #     # ax.set_ylim(0, self.ccd.horz_dimension)
    #     # ax.set_zlim(-1*self.ccd.total_thickness, 0)
    #     # ax.set_xlabel('vertical ($\mu$m)')
    #     # ax.set_ylabel('horizontal ($\mu$m)')
    #     # ax.set_zlabel('z ($\mu$m)')
    #     #
    #     # ax2.set_xlim(0, self.ccd.vert_dimension)
    #     # ax2.set_ylim(0, self.ccd.horz_dimension)
    #     # ax2.set_zlim(-1*self.ccd.total_thickness, 0)
    #     # ax2.set_xlabel('vertical ($\mu$m)')
    #     # ax2.set_ylabel('horizontal ($\mu$m)')
    #     # ax2.set_zlabel('z ($\mu$m)')
    #     ##################################
    #     # plt.show()

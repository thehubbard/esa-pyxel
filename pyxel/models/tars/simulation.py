#   --------------------------------------------------------------------------
#   Copyright 2018 SCI-FIV, ESA (European Space Agency)
#   --------------------------------------------------------------------------
"""PyXel! Simulation code for TARS model to generate charges by ionization."""

import typing as t  # noqa: F401
import numpy as np
from pyxel.models.tars.particle import Particle
from pyxel.models.tars.util import sampling_distribution, get_yvalue_with_interpolation
from pyxel.detectors.detector import Detector


class Simulation:
    """Main class of the program, Simulation contain all the methods to set and run a simulation."""

    def __init__(self, detector: Detector) -> None:
        """Initialize the simulation.

        :param Detector detector: Detector object(from CCD/CMSO library) containing all the simulated detector specs
        """
        self.detector = detector

        self.flux_dist = None
        self.spectrum_cdf = None

        self.energy_loss_data = None

        self.step_size_dist = None
        self.step_cdf = np.zeros((1, 2))

        self.let_dist = None
        self.let_cdf = np.zeros((1, 2))

        self.stopping_power = None

        self.particle_type = None
        self.initial_energy = None
        self.position_ver = None
        self.position_hor = None
        self.position_z = None
        self.angle_alpha = None
        self.angle_beta = None
        self.step_length = 1.0          # fix, all the other data/parameters should be adjusted to this
        self.energy_cut = 1.0e-5        # MeV

        self.e_num_lst = []     # type: t.List[int]
        self.e_energy_lst = []  # type: t.List[float]
        self.e_pos0_lst = []    # type: t.List[float]
        self.e_pos1_lst = []    # type: t.List[float]
        self.e_pos2_lst = []    # type: t.List[float]
        self.e_vel0_lst = []    # type: t.List[float]
        self.e_vel1_lst = []    # type: t.List[float]
        self.e_vel2_lst = []    # type: t.List[float]

        self.edep_per_step = []             # type: t.List[float]
        self.total_edep_per_particle = []   # type: t.List[float]

    def parameters(self, part_type, init_energy, pos_ver, pos_hor, pos_z, alpha, beta):
        """TBW.

        :param part_type:
        :param init_energy:
        :param pos_ver:
        :param pos_hor:
        :param pos_z:
        :param alpha:
        :param beta:
        :return:
        """
        self.particle_type = part_type
        self.initial_energy = init_energy
        self.position_ver = pos_ver
        self.position_hor = pos_hor
        self.position_z = pos_z
        self.angle_alpha = alpha
        self.angle_beta = beta

    # TODO
    def select_let(self, init_energy, det_thickness):
        """Select LET data which is relevant here before sampling it.

        Execute this for each new particle based on its initial energy (from
        input spectrum) and track length inside the detector.

        :param init_energy:
        :param det_thickness:
        :return:
        :warning NOT IMPLEMENTED:
        """
        pass

    def event_generation(self):
        """Generate an event on the CCD.

        The event occurs due to an incident particle taken according to the
        simulation configuration file.

        :return:
        """
        track_left = False
        geo = self.detector.geometry
        ioniz_energy = geo.material_ionization_energy

        particle = Particle(self.detector,
                            self.particle_type,
                            self.initial_energy, self.spectrum_cdf,
                            self.position_ver, self.position_hor, self.position_z,
                            self.angle_alpha, self.angle_beta)

        # TODO calculate total track length here with a new func
        track_len = particle.track_length()

        if self.energy_loss_data == 'let':
            self.select_let(particle.energy, self.detector.geometry.total_thickness)  # TODO

        # TODO implement select_stepsize_data func
        # if self.energy_loss_data == 'stepsize':
        #     self.select_stepsize_data(particle.energy, self.detector.geometry.total_thickness)

        # initial step, ionization can not happen in this first step
        particle.position[0] += particle.dir_ver * 0.01     # um
        particle.position[1] += particle.dir_hor * 0.01     # um
        particle.position[2] += particle.dir_z * 0.01       # um

        while True:
            # # check if p is still inside detector and have enough energy:
            # if particle.position[0] < 0.0 or particle.position[0] > self.detector.geometry.vert_dimension:
            #     break
            # if particle.position[1] < 0.0 or particle.position[1] > self.detector.geometry.horz_dimension:
            #     break
            # if particle.position[2] < -1 * self.detector.geometry.total_thickness or particle.position[2] > 0.0:
            #     break
            if particle.energy <= self.energy_cut:
                break
            #
            # track_left = True

            # IONIZATION
            # self._ionization_by_step_(p)
            #################################
            # geo = self.detector.geometry
            # ioniz_energy = geo.material_ionization_energy

            # particle.energy is in MeV !
            # particle.deposited_energy is in keV !
            if self.energy_loss_data == 'stepsize':
                current_step_size = sampling_distribution(self.let_cdf)  # um      # TODO change let_cdf to step_cdf

            particle.deposited_energy = 1.0  # keV       # TODO update this

            # UPDATE POSITION OF IONIZING PARTICLES
            particle.position[0] += particle.dir_ver * current_step_size  # um
            particle.position[1] += particle.dir_hor * current_step_size  # um
            particle.position[2] += particle.dir_z * current_step_size  # um

            # TODO if it is still inside the detector vol then ionization happen otherwise return 
            # check if p is still inside detector and have enough energy:
            if particle.position[0] < 0.0 or particle.position[0] > geo.vert_dimension:
                break
            if particle.position[1] < 0.0 or particle.position[1] > geo.horz_dimension:
                break
            if particle.position[2] < -1 * geo.total_thickness or particle.position[2] > 0.0:
                break
            if particle.deposited_energy >= particle.energy * 1e3:
                # particle.deposited_energy = particle.energy * 1e3
                break

            track_left = True

            # keV
            # particle.deposited_energy = particle.electrons * (e_kin_energy + ioniz_energy) * 1e-3
            particle.energy -= particle.deposited_energy * 1e-3  # MeV

            e_kin_energy = 1000.0  # eV     # TODO sample kinetic energy from data

            # particle.electrons = int(particle.deposited_energy * 1e3 / (ioniz_energy + e_kin_energy))  # eV/eV = 1
            particle.electrons = 1
            # TODO implement clusters here

            self.e_num_lst += [particle.electrons]
            self.e_energy_lst += [e_kin_energy]
            self.e_pos0_lst += [particle.position[0]]
            self.e_pos1_lst += [particle.position[1]]
            self.e_pos2_lst += [particle.position[2]]

            self.edep_per_step.append(particle.deposited_energy)  # keV
            particle.total_edep += particle.deposited_energy  # keV
            #################################

            # save particle trajectory
            particle.trajectory = np.vstack((particle.trajectory, particle.position))
        # END of loop

        if track_left:
            self.total_edep_per_particle.append(particle.total_edep)  # keV

    # TODO: make two different function using let or stopping power
    def _ionization_by_step_(self, particle):
        """TBW.

        :param particle:
        :return:
        """
        geo = self.detector.geometry
        ioniz_energy = geo.material_ionization_energy

        # particle.energy is in MeV !
        # particle.deposited_energy is in keV !
        if self.energy_loss_data == 'stepsize':
            current_step_size = sampling_distribution(self.let_cdf)  # um      # TODO change let_cdf to step_cdf
        else:
            return

        # UPDATE POSITION OF IONIZING PARTICLES
        particle.position[0] += particle.dir_ver * current_step_size  # um
        particle.position[1] += particle.dir_hor * current_step_size  # um
        particle.position[2] += particle.dir_z * current_step_size    # um

        # TODO if it is still inside the detector vol then ionization happen otherwise return 
        # if 
        
        
        
        particle.deposited_energy = 1.0     # keV       # TODO update this

        if particle.deposited_energy >= particle.energy * 1e3:
            particle.deposited_energy = particle.energy * 1e3

        e_kin_energy = 1000.0  # eV     # TODO sample kinetic energy from data

        # particle.electrons = int(particle.deposited_energy * 1e3 / (ioniz_energy + e_kin_energy))  # eV/eV = 1
        particle.electrons = 1
        # TODO implement clusters here

        self.e_num_lst += [particle.electrons]
        self.e_energy_lst += [e_kin_energy]
        self.e_pos0_lst += [particle.position[0]]
        self.e_pos1_lst += [particle.position[1]]
        self.e_pos2_lst += [particle.position[2]]

        # keV
        particle.deposited_energy = particle.electrons * (e_kin_energy + ioniz_energy) * 1e-3
        particle.energy -= particle.deposited_energy * 1e-3     # MeV

        self.edep_per_step.append(particle.deposited_energy)    # keV
        particle.total_edep += particle.deposited_energy        # keV

    # TODO: make two different function using let or stopping power
    def _ionization_(self, particle):
        """TBW.

        :param particle:
        :return:
        """
        geo = self.detector.geometry
        ioniz_energy = geo.material_ionization_energy
        let_value = None

        # particle.energy is in MeV !
        # particle.deposited_energy is in keV !
        if self.energy_loss_data == 'let':
            let_value = sampling_distribution(self.let_cdf)  # keV/um
        elif self.energy_loss_data == 'stopping':
            stopping_power = get_yvalue_with_interpolation(self.stopping_power, particle.energy)  # MeV*cm2/g
            let_value = 0.1 * stopping_power * geo.material_density  # keV/um

        particle.deposited_energy = let_value * self.step_length  # keV

        if particle.deposited_energy >= particle.energy * 1e3:
            particle.deposited_energy = particle.energy * 1e3

        e_kin_energy = 0.1  # eV
        particle.electrons = int(particle.deposited_energy * 1e3 / (ioniz_energy + e_kin_energy))  # eV/eV = 1

        self.e_num_lst += [particle.electrons]
        self.e_energy_lst += [e_kin_energy]
        self.e_pos0_lst += [particle.position[0]]
        self.e_pos1_lst += [particle.position[1]]
        self.e_pos2_lst += [particle.position[2]]

        # keV
        particle.deposited_energy = particle.electrons * (e_kin_energy + ioniz_energy) * 1e-3
        particle.energy -= particle.deposited_energy * 1e-3     # MeV

        self.edep_per_step.append(particle.deposited_energy)    # keV
        particle.total_edep += particle.deposited_energy        # keV

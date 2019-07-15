#   --------------------------------------------------------------------------
#   Copyright 2019 SCI-FIV, ESA (European Space Agency)
#   --------------------------------------------------------------------------
"""Pyxel Charge class to generate electrons or holes inside detector."""
import pandas as pd
import numpy as np
from astropy.units import cds
from pyxel.data_structure.particle import Particle
import typing as t

cds.enable()


class Charge(Particle):
    """Charged particle class defining and storing information of all electrons and holes.

    Properties stored are: charge, position, velocity, energy.
    """

    def __init__(self):
        """TBW."""
        # FRED: The following line is not really needed
        super().__init__()
        self.nextid = 0  # type: int

        # FRED: This could be a `Tuple` (because of immutability) ?
        self.columns = ['charge',
                        'number',
                        'init_energy', 'energy',
                        'init_pos_ver', 'init_pos_hor', 'init_pos_z',
                        'position_ver', 'position_hor', 'position_z',
                        'velocity_ver', 'velocity_hor', 'velocity_z']  # type: t.List[str]

        self.EMPTY_FRAME = pd.DataFrame(columns=self.columns,
                                        dtype=np.float)   # type: pd.DataFrame       # todo

        self.frame = self.EMPTY_FRAME.copy()  # type: pd.DataFrame

    def add_charge(self,
                   particle_type: str,
                   particles_per_cluster,
                   init_energy: t.List[float],
                   init_ver_position: t.List[float],
                   init_hor_position: t.List[float],
                   init_z_position: t.List[float],
                   init_ver_velocity: t.List[float],
                   init_hor_velocity: t.List[float],
                   init_z_velocity: t.List[float]) -> None:
        """Create new charge or group of charge inside the detector stored in a pandas DataFrame.

        :param particle_type:
        :param particles_per_cluster:
        :param init_energy:
        :param init_ver_position:
        :param init_hor_position:
        :param init_z_position:
        :param init_ver_velocity:
        :param init_hor_velocity:
        :param init_z_velocity:
        :return:
        """
        if len(particles_per_cluster) == len(init_energy) == len(init_ver_position) == len(init_ver_velocity):
            elements = len(init_energy)
        else:
            raise ValueError('List arguments have different lengths')

        # check_position(self.detector, init_ver_position, init_hor_position, init_z_position)      # TODO
        # check_energy(init_energy)             # TODO
        # Check if particle number is integer:
        # check_type(particles_per_cluster)      # TODO

        # FRED: particle_type should be a Enum class ?
        if particle_type == 'e':
            charge = [-1] * elements            # * cds.e
        elif particle_type == 'h':
            charge = [+1] * elements            # * cds.e
        else:
            raise ValueError('Given charged particle type can not be simulated')

        # if all(init_ver_velocity) == 0 and all(init_hor_velocity) == 0 and all(init_z_velocity) == 0:
        #     random_direction(1.0)

        # dict
        new_charge = {'charge': charge,
                      'number': particles_per_cluster,
                      'init_energy': init_energy,
                      'energy': init_energy,
                      'init_pos_ver': init_ver_position,
                      'init_pos_hor': init_hor_position,
                      'init_pos_z': init_z_position,
                      'position_ver': init_ver_position,
                      'position_hor': init_hor_position,
                      'position_z': init_z_position,
                      'velocity_ver': init_ver_velocity,
                      'velocity_hor': init_hor_velocity,
                      'velocity_z': init_z_velocity}  # type: dict

        new_charge_df = pd.DataFrame(new_charge, index=range(self.nextid, self.nextid + elements))
        self.nextid = self.nextid + elements
        self.frame = self.frame.append(new_charge_df, sort=False)

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Pyxel Charge class to generate electrons or holes inside detector."""

import typing as t

import numpy as np
import pandas as pd

from pyxel.data_structure import Particle


class Charge(Particle):
    """Charged particle class defining and storing information of all electrons and holes.

    Properties stored are: charge, position, velocity, energy.
    """

    def __init__(self):
        # TODO: The following line is not really needed
        super().__init__()
        self.nextid = 0  # type: int

        self.columns = (
            "charge",
            "number",
            "init_energy",
            "energy",
            "init_pos_ver",
            "init_pos_hor",
            "init_pos_z",
            "position_ver",
            "position_hor",
            "position_z",
            "velocity_ver",
            "velocity_hor",
            "velocity_z",
        )  # type: t.Tuple[str, ...]

        self.EMPTY_FRAME = pd.DataFrame(
            columns=self.columns, dtype=float
        )  # type: pd.DataFrame

        self.frame = self.EMPTY_FRAME.copy()  # type: pd.DataFrame

    @staticmethod
    def create_charges(
        *,
        particle_type: str,  # TODO: Use Enum
        particles_per_cluster: np.ndarray,
        init_energy: np.ndarray,
        init_ver_position: np.ndarray,
        init_hor_position: np.ndarray,
        init_z_position: np.ndarray,
        init_ver_velocity: np.ndarray,
        init_hor_velocity: np.ndarray,
        init_z_velocity: np.ndarray,
    ) -> pd.DataFrame:
        """Create new charge(s) or group of charge(s) as a `DataFrame`.

        Parameters
        ----------
        particle_type : str
            Type of particle. Valid values: 'e' for an electron or 'h' for a hole.
        particles_per_cluster : array-like
        init_energy : array-like
        init_ver_position : array-like
        init_hor_position : array-like
        init_z_position : array-like
        init_ver_velocity : array-like
        init_hor_velocity : array-like
        init_z_velocity : array-like

        Returns
        -------
        dataframe
            Charge(s) stored in a `DataFrame`.
        """
        if not (
            len(particles_per_cluster)
            == len(init_energy)
            == len(init_ver_position)
            == len(init_hor_position)
            == len(init_z_position)
            == len(init_ver_velocity)
            == len(init_hor_velocity)
            == len(init_z_velocity)
        ):
            raise ValueError("List arguments have different lengths.")

        if not (
            particles_per_cluster.ndim
            == init_energy.ndim
            == init_ver_position.ndim
            == init_hor_position.ndim
            == init_z_position.ndim
            == init_ver_velocity.ndim
            == init_hor_velocity.ndim
            == init_z_velocity.ndim
            == 1
        ):
            raise ValueError("List arguments must have only one dimension.")

        elements = len(init_energy)

        # check_position(self.detector, init_ver_position, init_hor_position, init_z_position)      # TODO
        # check_energy(init_energy)             # TODO
        # Check if particle number is integer:
        # check_type(particles_per_cluster)      # TODO

        # TODO: particle_type should be a Enum class ?
        if particle_type == "e":
            charge = [-1] * elements  # * cds.e
        elif particle_type == "h":
            charge = [+1] * elements  # * cds.e
        else:
            raise ValueError("Given charged particle type can not be simulated")

        # if all(init_ver_velocity) == 0 and all(init_hor_velocity) == 0 and all(init_z_velocity) == 0:
        #     random_direction(1.0)

        # Create new charges as a `dict`
        new_charges = {
            "charge": charge,
            "number": particles_per_cluster,
            "init_energy": init_energy,
            "energy": init_energy,
            "init_pos_ver": init_ver_position,
            "init_pos_hor": init_hor_position,
            "init_pos_z": init_z_position,
            "position_ver": init_ver_position,
            "position_hor": init_hor_position,
            "position_z": init_z_position,
            "velocity_ver": init_ver_velocity,
            "velocity_hor": init_hor_velocity,
            "velocity_z": init_z_velocity,
        }  # type: t.Mapping[str, t.Union[t.Sequence, np.ndarray]]

        return pd.DataFrame(new_charges)

    def add_charge_dataframe(self, new_charges: pd.DataFrame) -> None:
        """Add new charge(s) or group of charge(s) inside the detector.

        Parameters
        ----------
        new_charges : DataFrame
            Charges as a `DataFrame`
        """
        if set(new_charges.columns) != set(self.columns):
            expected_columns = ", ".join(map(repr, self.columns))  # type: str
            raise ValueError(f"Expected columns: {expected_columns}")

        if self.frame.empty:
            new_frame = new_charges  # type: pd.DataFrame
        else:
            new_frame = self.frame.append(new_charges, ignore_index=True)

        self.frame = new_frame
        self.nextid = self.nextid + len(new_charges)

    def add_charge(
        self,
        *,
        particle_type: str,  # TODO: Use Enum
        particles_per_cluster: np.ndarray,
        init_energy: np.ndarray,
        init_ver_position: np.ndarray,
        init_hor_position: np.ndarray,
        init_z_position: np.ndarray,
        init_ver_velocity: np.ndarray,
        init_hor_velocity: np.ndarray,
        init_z_velocity: np.ndarray,
    ) -> None:
        """Add new charge(s) or group of charge(s) inside the detector.

        Parameters
        ----------
        particle_type : str
            Type of particle. Valid values: 'e' for an electron or 'h' for a hole.
        particles_per_cluster : array-like
        init_energy : array-like
        init_ver_position : array-like
        init_hor_position : array-like
        init_z_position : array-like
        init_ver_velocity : array-like
        init_hor_velocity : array-like
        init_z_velocity : array-like
        """
        # Create charge(s)
        new_charges = Charge.create_charges(
            particle_type=particle_type,
            particles_per_cluster=particles_per_cluster,
            init_energy=init_energy,
            init_ver_position=init_ver_position,
            init_hor_position=init_hor_position,
            init_z_position=init_z_position,
            init_ver_velocity=init_ver_velocity,
            init_hor_velocity=init_hor_velocity,
            init_z_velocity=init_z_velocity,
        )  # type: pd.DataFrame

        # Add charge(s)
        self.add_charge_dataframe(new_charges=new_charges)

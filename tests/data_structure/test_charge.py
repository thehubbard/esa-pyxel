#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import typing as t
from dataclasses import dataclass

import pandas as pd
import pytest

from pyxel.data_structure import Charge


def test_empty_charge():
    """Test an empty `Charge` object."""
    charge = Charge()

    assert charge.nextid == 0

    pd.testing.assert_frame_equal(
        charge.frame,
        pd.DataFrame(
            data=None,
            columns=(
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
            ),
            dtype=float,
        ),
    )


def test_add_one_charge():
    """Test with one charge."""
    charge = Charge()

    charge.add_charge(
        particle_type="e",
        particles_per_cluster=[1],
        init_energy=[0.1],
        init_ver_position=[1.1],
        init_hor_position=[2.2],
        init_z_position=[3.3],
        init_ver_velocity=[4.4],
        init_hor_velocity=[-5.5],
        init_z_velocity=[6.6],
    )

    exp_df_charges = pd.DataFrame(
        {
            "charge": [-1.0],
            "number": [1.0],
            "init_energy": [0.1],
            "energy": [0.1],
            "init_pos_ver": [1.1],
            "init_pos_hor": [2.2],
            "init_pos_z": [3.3],
            "position_ver": [1.1],
            "position_hor": [2.2],
            "position_z": [3.3],
            "velocity_ver": [4.4],
            "velocity_hor": [-5.5],
            "velocity_z": [6.6],
        }
    )

    assert charge.nextid == 1
    pd.testing.assert_frame_equal(charge.frame, exp_df_charges)


def test_add_one_hole():
    """Test with one hole."""
    charge = Charge()

    charge.add_charge(
        particle_type="h",
        particles_per_cluster=[1],
        init_energy=[0.1],
        init_ver_position=[1.1],
        init_hor_position=[2.2],
        init_z_position=[3.3],
        init_ver_velocity=[4.4],
        init_hor_velocity=[-5.5],
        init_z_velocity=[6.6],
    )

    exp_df_charges = pd.DataFrame(
        {
            "charge": [1.0],
            "number": [1.0],
            "init_energy": [0.1],
            "energy": [0.1],
            "init_pos_ver": [1.1],
            "init_pos_hor": [2.2],
            "init_pos_z": [3.3],
            "position_ver": [1.1],
            "position_hor": [2.2],
            "position_z": [3.3],
            "velocity_ver": [4.4],
            "velocity_hor": [-5.5],
            "velocity_z": [6.6],
        }
    )

    assert charge.nextid == 1
    pd.testing.assert_frame_equal(charge.frame, exp_df_charges)


@dataclass
class ChargeInfo:
    """Define one charge.

    Only for testing.
    """

    particle_type: str
    particles_per_cluster: t.Sequence[int]
    init_energy: t.Sequence[float]
    init_ver_position: t.Sequence[float]
    init_hor_position: t.Sequence[float]
    init_z_position: t.Sequence[float]
    init_ver_velocity: t.Sequence[float]
    init_hor_velocity: t.Sequence[float]
    init_z_velocity: t.Sequence[float]


@pytest.mark.parametrize(
    "param_name, new_value, exp_exc, exp_error",
    [
        # Wrong parameter 'particle_type'
        pytest.param(
            "particle_type",
            "E",
            ValueError,
            "Given charged particle type can not be simulated",
            id="Wrong 'particle_type': 'E'",
        ),
        pytest.param(
            "particle_type",
            "H",
            ValueError,
            "Given charged particle type can not be simulated",
            id="Wrong 'particle_type': 'H'",
        ),
        pytest.param(
            "particle_type",
            "electron",
            ValueError,
            "Given charged particle type can not be simulated",
            id="Wrong 'particle_type': 'electron'",
        ),
        pytest.param(
            "particle_type",
            "hole",
            ValueError,
            "Given charged particle type can not be simulated",
            id="Wrong 'particle_type': 'hole'",
        ),
        # Wrong parameter 'particles_per_cluster'
        pytest.param(
            "particles_per_cluster",
            [],
            ValueError,
            "List arguments have different lengths",
            id="Too little 'particles_per_cluster'",
        ),
        pytest.param(
            "particles_per_cluster",
            [1, 2],
            ValueError,
            "List arguments have different lengths",
            id="Too many 'particles_per_cluster'",
        ),
    ],
)
def test_invalid_add_charge(
    param_name: str, new_value, exp_exc: Exception, exp_error: str
):
    """Test method `Charge.add_charge` with invalid parameters."""
    charge = Charge()

    # Create valid parameters
    params = ChargeInfo(
        particle_type="e",
        particles_per_cluster=[1],
        init_energy=[0.1],
        init_ver_position=[1.1],
        init_hor_position=[2.2],
        init_z_position=[3.3],
        init_ver_velocity=[4.4],
        init_hor_velocity=[-5.5],
        init_z_velocity=[6.6],
    )

    # Add the valid parameters
    charge.add_charge(
        particle_type=params.particle_type,
        particles_per_cluster=params.particles_per_cluster,
        init_energy=params.init_energy,
        init_ver_position=params.init_ver_position,
        init_hor_position=params.init_hor_position,
        init_z_position=params.init_z_position,
        init_ver_velocity=params.init_ver_velocity,
        init_hor_velocity=params.init_hor_velocity,
        init_z_velocity=params.init_z_velocity,
    )

    # Add an invalid parameter
    # This is equivalent to 'params.param_name = new_value'
    setattr(params, param_name, new_value)

    with pytest.raises(exp_exc, match=exp_error):
        charge.add_charge(
            particle_type=params.particle_type,
            particles_per_cluster=params.particles_per_cluster,
            init_energy=params.init_energy,
            init_ver_position=params.init_ver_position,
            init_hor_position=params.init_hor_position,
            init_z_position=params.init_z_position,
            init_ver_velocity=params.init_ver_velocity,
            init_hor_velocity=params.init_hor_velocity,
            init_z_velocity=params.init_z_velocity,
        )


def test_add_two_charges():
    """Test when adding one charges in one time."""
    charge = Charge()

    charge.add_charge(
        particle_type="e",
        particles_per_cluster=[1, 2],
        init_energy=[0.1, 0.2],
        init_ver_position=[1.11, 1.12],
        init_hor_position=[2.21, 2.22],
        init_z_position=[3.31, 3.32],
        init_ver_velocity=[4.41, 4.42],
        init_hor_velocity=[-5.51, 5.52],
        init_z_velocity=[6.61, 6.62],
    )

    exp_df_charges = pd.DataFrame(
        {
            "charge": [-1.0, -1.0],
            "number": [1.0, 2.0],
            "init_energy": [0.1, 0.2],
            "energy": [0.1, 0.2],
            "init_pos_ver": [1.11, 1.12],
            "init_pos_hor": [2.21, 2.22],
            "init_pos_z": [3.31, 3.32],
            "position_ver": [1.11, 1.12],
            "position_hor": [2.21, 2.22],
            "position_z": [3.31, 3.32],
            "velocity_ver": [4.41, 4.42],
            "velocity_hor": [-5.51, 5.52],
            "velocity_z": [6.61, 6.62],
        }
    )

    assert charge.nextid == 2
    pd.testing.assert_frame_equal(charge.frame, exp_df_charges)


def test_add_two_charges_one_hole():
    """Test when adding two charges and then one hole."""
    charge = Charge()

    # Add 2 charges
    charge.add_charge(
        particle_type="e",
        particles_per_cluster=[1, 2],
        init_energy=[0.1, 0.2],
        init_ver_position=[1.11, 1.12],
        init_hor_position=[2.21, 2.22],
        init_z_position=[3.31, 3.32],
        init_ver_velocity=[4.41, 4.42],
        init_hor_velocity=[-5.51, 5.52],
        init_z_velocity=[6.61, 6.62],
    )

    # Add 1 hole
    charge.add_charge(
        particle_type="h",
        particles_per_cluster=[3],
        init_energy=[0.3],
        init_ver_position=[1.13],
        init_hor_position=[2.23],
        init_z_position=[3.33],
        init_ver_velocity=[4.43],
        init_hor_velocity=[5.53],
        init_z_velocity=[6.63],
    )

    exp_df_charges = pd.DataFrame(
        {
            "charge": [-1.0, -1.0, 1],
            "number": [1.0, 2.0, 3],
            "init_energy": [0.1, 0.2, 0.3],
            "energy": [0.1, 0.2, 0.3],
            "init_pos_ver": [1.11, 1.12, 1.13],
            "init_pos_hor": [2.21, 2.22, 2.23],
            "init_pos_z": [3.31, 3.32, 3.33],
            "position_ver": [1.11, 1.12, 1.13],
            "position_hor": [2.21, 2.22, 2.23],
            "position_z": [3.31, 3.32, 3.33],
            "velocity_ver": [4.41, 4.42, 4.43],
            "velocity_hor": [-5.51, 5.52, 5.53],
            "velocity_z": [6.61, 6.62, 6.63],
        }
    )

    assert charge.nextid == 3
    pd.testing.assert_frame_equal(charge.frame, exp_df_charges)

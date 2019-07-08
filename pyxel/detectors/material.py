"""Geometry class for detector."""
import typing as t
import numpy as np
import esapy_config.config as ec
from esapy_config import validators
import os


# FRED: Fix this
check_path = os.path.exists


# Universal global constants
M_ELECTRON = 9.10938356e-31    # kg     # TODO put these global constants to a data file


@ec.config(mode='RO')
class Material:
    """Material attributes of the detector."""

    # TODO create func for compound materials
    # def set_material(self, material):
    #     """Set material properties.
    #
    #     :param material:
    #     """
    #     # TODO put these constants to a data file
    #     if material == 'silicon' or 'Si' or 'si':
    #         self.material_density = 2.328                 # (g/cm3)
    #         self.ionization_energy = 3.6                  # (eV)
    #         self.band_gap = 1.12                          # (eV)
    #         self.e_effective_mass = 0.5 * M_ELECTRON      # (kg)
    #
    #     else:
    #         raise NotImplementedError('Given material has not implemented yet')

    # def __attrs_post_init__(self):
    #     """Setting material."""
    #     if self.material:
    #         self.set_material(self.material)

    # FRED: What is it ?
    def load_numpy_array(self, attr=None, path=None):
        """Create Numpy array storing data temporarily."""
        if isinstance(path, str):
            if check_path(path):
                if path.endswith('.npy'):
                    setattr(self, '_' + attr, np.load(path))

    trapped_charge = ec.setting(
        type=t.Optional[str],
        default=None,
        # on_change=load_numpy_array,  # FRED: What is it ?
        # validator=[(pyx.validate_type(str) or pyx.validate_type(np.ndarray))], <<< this does not work
        doc='Numpy array storing the trap density temporarily'
    )

    n_acceptor = ec.setting(
        type=float,
        default=0.0,
        validator=validators.interval(0., 1000.),
        metadata={'units': 'cm-3'},
        doc='Density of acceptors in the lattice'
    )
    n_donor = ec.setting(
        type=float,
        default=0.0,
        validator=validators.interval(0., 1000.),
        metadata={'units': 'cm-3'},
        doc='Density of donors in the lattice'
    )
    material = ec.setting(
        type=str,
        default='silicon',
        validator=validators.validate_in(['silicon', 'hxrg']),
        # on_set=set_material,
        doc='Semiconductor material of the detector'
    )
    material_density = ec.setting(       # todo: set automatically depending on the material
        # init=False,
        type=float,
        default=2.328,                      # Silicon
        validator=validators.interval(0., 10000.),
        metadata={'units': 'g/cm3'},
        doc='Material density'
    )
    ionization_energy = ec.setting(       # todo: set automatically depending on the material
        # init=False,
        type=float,
        default=3.6,                        # Silicon
        validator=validators.interval(0., 100.),
        metadata={'units': 'eV'},
        doc='Mean ionization energy of the semiconductor lattice'
    )
    band_gap = ec.setting(       # todo: set automatically depending on the material
        # init=False,
        type=float,
        default=1.12,                       # Silicon
        validator=validators.interval(0., 10.),
        metadata={'units': 'eV'},
        doc='Band gap of the semiconductor lattice'
    )
    e_effective_mass = ec.setting(       # todo: set automatically depending on the material
        # init=False,
        type=float,
        default=0.5 * M_ELECTRON,           # Silicon
        validator=validators.interval(0., 1.e-10),
        metadata={'units': 'kg'},
        doc='Electron effective mass in the semiconductor lattice'
    )

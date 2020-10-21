#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Pyxel general particle class to track particles like photon, electrons, holes."""

import typing as t

import numpy as np
import pandas as pd

from pyxel.util.memory import get_size

# from astropy.units import cds
# cds.enable()


class Particle:
    """Class defining and storing information of all particles with their position, velocity, energy, etc."""

    def __init__(self):
        # TODO: This should be a class variable
        self.EMPTY_FRAME = pd.DataFrame()  # type: pd.DataFrame
        self.frame = pd.DataFrame()  # type: pd.DataFrame

        self._numbytes = 0

    def get_values(self, quantity: str, id_list: t.Optional[list] = None) -> np.ndarray:
        """Get quantity values of particles defined with id_list. By default it returns values of all particles.

        :param quantity: name of quantity: ``number``, ``energy``, ``position_ver``, ``velocity_hor``, etc.
        :param id_list: list of particle ids: ``[0, 12, 321]``
        :return: array
        """
        if id_list:
            array = self.frame.query("index in %s" % id_list)[quantity].values
        else:
            array = self.frame[quantity].values
        return array

    def set_values(
        self, quantity: str, new_value_list: list, id_list: t.Optional[list] = None
    ) -> None:
        """Update quantity values of particles defined with id_list. By default it updates all.

        :param quantity: name of quantity: ``number``, ``energy``, ``position_ver``, ``velocity_hor``, etc.
        :param new_value_list: list of values ``[1.12, 2.23, 3.65]``
        :param id_list: list of particle ids: ``[0, 12, 321]``
        """
        new_df = pd.DataFrame({quantity: new_value_list}, index=id_list)
        self.frame.update(new_df)

    def remove(self, id_list: t.Optional[list] = None) -> None:
        """Remove particles defined with id_list. By default it removes all particles from DataFrame.

        :param id_list: list of particle ids: ``[0, 12, 321]``
        """
        if id_list:
            # TODO: Check carefully if 'inplace' is needed. This could break lot of things.
            self.frame.query("index not in %s" % id_list, inplace=True)
        else:
            self.frame = self.EMPTY_FRAME.copy()

    @property
    def numbytes(self) -> int:
        """Recursively calculates object size in bytes using Pympler library.

        Returns
        -------
        int
            Size of the object in bytes.
        """
        self._numbytes = get_size(self)
        return self._numbytes

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""MKID-array detector modeling class."""

import typing as t

from pyxel.data_structure import Phase
from pyxel.detectors import Detector

if t.TYPE_CHECKING:
    from pyxel.detectors import Environment, Material, MKIDCharacteristics, MKIDGeometry


class MKID(Detector):
    """MKID-based detector class containing all detector attributes and data."""

    def __init__(
        self,
        geometry: "MKIDGeometry",
        material: "Material",
        environment: "Environment",
        characteristics: "MKIDCharacteristics",
    ):
        self._geometry = geometry  # type: MKIDGeometry
        self._characteristics = characteristics  # type: MKIDCharacteristics
        self._phase = None  # type: t.Optional[Phase]

        super().__init__(material=material, environment=environment)
        super().reset()

    def reset(self, reset_all: bool = True) -> None:
        """TBW."""
        super().reset(reset_all=reset_all)

        if reset_all:
            self._phase = Phase(self.geometry)

    @property
    def geometry(self) -> "MKIDGeometry":
        """TBW."""
        return self._geometry

    @property
    def characteristics(self) -> "MKIDCharacteristics":
        """TBW."""
        return self._characteristics

    @property
    def phase(self) -> Phase:
        """TBW."""
        if not self._phase:
            raise RuntimeError("'phase' not initialized.")

        return self._phase

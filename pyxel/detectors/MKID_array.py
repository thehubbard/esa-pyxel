#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""MKID-array detector modeling class."""

import typing as t

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

        super().__init__(material=material, environment=environment)
        super().initialize()

    @property
    def geometry(self) -> "MKIDGeometry":
        """TBW."""
        return self._geometry

    @property
    def characteristics(self) -> "MKIDCharacteristics":
        """TBW."""
        return self._characteristics

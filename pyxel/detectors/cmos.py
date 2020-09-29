#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""CMOS detector modeling class."""
import typing as t

from pyxel.detectors import Detector

if t.TYPE_CHECKING:
    from pyxel.detectors import CMOSCharacteristics, CMOSGeometry, Environment, Material


class CMOS(Detector):
    """CMOS-based detector class containing all detector attributes and data."""

    def __init__(
        self,
        geometry: "CMOSGeometry",
        material: "Material",
        environment: "Environment",
        characteristics: "CMOSCharacteristics",
    ):
        self._geometry = geometry  # type: CMOSGeometry
        self._characteristics = characteristics  # type: CMOSCharacteristics

        super().__init__(material=material, environment=environment)
        super().initialize()

    @property
    def geometry(self) -> "CMOSGeometry":
        """TBW."""
        return self._geometry

    @property
    def characteristics(self) -> "CMOSCharacteristics":
        """TBW."""
        return self._characteristics

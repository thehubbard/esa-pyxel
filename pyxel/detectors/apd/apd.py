#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

""":term:`CMOS` detector modeling class."""
import typing as t

from pyxel.detectors import Detector

if t.TYPE_CHECKING:
    from pyxel.detectors import Environment

    from .apd_characteristics import APDCharacteristics
    from .apd_geometry import APDGeometry


class APD(Detector):
    """:term:`CMOS`-based detector class containing all detector attributes and data."""

    def __init__(
        self,
        geometry: "APDGeometry",
        environment: "Environment",
        characteristics: "APDCharacteristics",
    ):
        self._geometry = geometry  # type: APDGeometry
        self._characteristics = characteristics  # type: APDCharacteristics

        super().__init__(environment=environment)
        super().reset()

    @property
    def geometry(self) -> "APDGeometry":
        """TBW."""
        return self._geometry

    @property
    def characteristics(self) -> "APDCharacteristics":
        """TBW."""
        return self._characteristics

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

""":term:`CMOS` detector modeling class."""
import typing as t

from pyxel.detectors import (
    CMOSCharacteristics,
    CMOSGeometry,
    Detector,
    Environment,
)


class CMOS(Detector):
    """:term:`CMOS`-based detector class containing all detector attributes and data."""

    def __init__(
        self,
        geometry: CMOSGeometry,
        environment: Environment,
        characteristics: CMOSCharacteristics,
    ):
        self._geometry = geometry  # type: CMOSGeometry
        self._characteristics = characteristics  # type: CMOSCharacteristics

        super().__init__(environment=environment)
        super().reset()

    def __eq__(self, other) -> bool:
        return (
            type(self) == type(other)
            and self.geometry == other.geometry
            and self.environment == other.environment
            and self.characteristics == other.characteristics
        )

    @property
    def geometry(self) -> CMOSGeometry:
        """TBW."""
        return self._geometry

    @property
    def characteristics(self) -> CMOSCharacteristics:
        """TBW."""
        return self._characteristics

    def to_dict(self) -> dict:
        """Get the attributes of this instance as a `dict`."""
        dct = {
            "type": "cmos",
            "geometry": self.geometry.to_dict(),
            "environment": self.environment.to_dict(),
            "characteristics": self.characteristics.to_dict(),
        }

        return dct

    @classmethod
    def from_dict(cls, dct: dict):
        """Create a new instance of `CCD` from a `dict`."""
        # TODO: This is a simplistic implementation. Improve this.
        if dct["type"] != "cmos":
            raise ValueError

        geometry = CMOSGeometry.from_dict(dct["geometry"])
        environment = Environment.from_dict(dct["environment"])
        characteristics = CMOSCharacteristics.from_dict(dct["characteristics"])

        return cls(
            geometry=geometry,
            environment=environment,
            characteristics=characteristics,
        )

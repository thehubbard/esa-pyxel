#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""CCD detector modeling class."""

from pyxel.detectors import (
    CCDCharacteristics,
    CCDGeometry,
    Detector,
    Environment,
)


class CCD(Detector):
    """Charge-Coupled Device class containing all detector attributes and data."""

    def __init__(
        self,
        geometry: CCDGeometry,
        environment: Environment,
        characteristics: CCDCharacteristics,
    ):
        self._geometry = geometry  # type: CCDGeometry
        self._characteristics = characteristics  # type: CCDCharacteristics

        super().__init__(environment=environment)
        super().reset()

    def __eq__(self, other) -> bool:
        return (
            type(self) == type(other)
            and self.geometry == other.geometry
            and self.material == other.material
            and self.environment == other.environment
            and self.characteristics == other.characteristics
        )

    @property
    def geometry(self) -> CCDGeometry:
        """TBW."""
        return self._geometry

    @property
    def characteristics(self) -> CCDCharacteristics:
        """TBW."""
        return self._characteristics

    def to_dict(self) -> dict:
        """Get the attributes of this instance as a `dict`."""
        dct = {
            "type": "ccd",
            "geometry": self.geometry.to_dict(),
            "material": self.material.to_dict(),
            "environment": self.environment.to_dict(),
            "characteristics": self.characteristics.to_dict(),
            "arrays": {
                "photon": None if self._photon is None else self._photon.array.copy(),
                "pixel": None if self._pixel is None else self._pixel.array.copy(),
                "signal": None if self._signal is None else self._signal.array.copy(),
                "image": None if self._image is None else self._image.array.copy(),
            },
            "particles": {
                "charge": None if self._charge is None else self._charge.frame.copy()
            },
        }

        return dct

    @classmethod
    def from_dict(cls, dct: dict):
        """Create a new instance of `CCD` from a `dict`."""
        # TODO: This is a simplistic implementation. Improve this.
        if dct["type"] != "ccd":
            raise ValueError

        geometry = CCDGeometry.from_dict(dct["geometry"])
        material = Material.from_dict(dct["material"])
        environment = Environment.from_dict(dct["environment"])
        characteristics = CCDCharacteristics.from_dict(dct["characteristics"])

        return cls(
            geometry=geometry,
            material=material,
            environment=environment,
            characteristics=characteristics,
        )

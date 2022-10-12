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

""":term:`APD` detector modeling class."""

from typing import TYPE_CHECKING, Mapping, Optional

from pyxel.detectors import Detector

if TYPE_CHECKING:
    import pandas as pd

    from pyxel.detectors import APDCharacteristics, APDGeometry, Environment


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

    def __eq__(self, other) -> bool:
        return (
            type(self) == type(other)
            and self.geometry == other.geometry
            and self.environment == other.environment
            and self.characteristics == other.characteristics
            and super().__eq__(other)
        )

    @property
    def geometry(self) -> "APDGeometry":
        """TBW."""
        return self._geometry

    @property
    def characteristics(self) -> "APDCharacteristics":
        """TBW."""
        return self._characteristics

    # TODO: Refactor this
    def to_dict(self) -> Mapping:
        """Convert an instance of `APD` to a `dict`."""
        dct = {
            "version": 1,
            "type": "APD",
            "properties": {
                "geometry": self.geometry.to_dict(),
                "environment": self.environment.to_dict(),
                "characteristics": self.characteristics.to_dict(),
            },
            "data": {
                "photon": None if self._photon is None else self._photon.array.copy(),
                "scene": None if self._scene is None else self._scene.to_dict(),
                "pixel": None if self._pixel is None else self._pixel.array.copy(),
                "signal": None if self._signal is None else self._signal.array.copy(),
                "image": None if self._image is None else self._image.array.copy(),
                "charge": None
                if self._charge is None
                else {
                    "array": self._charge.array.copy(),
                    "frame": self._charge.frame.copy(),
                },
            },
        }

        return dct

    # TODO: Refactor this
    @classmethod
    def from_dict(cls, dct: Mapping) -> "APD":
        """Create a new instance of `APD` from a `dict`."""
        # TODO: This is a simplistic implementation. Improve this.
        import numpy as np

        from pyxel.data_structure import Scene
        from pyxel.detectors import APDCharacteristics, APDGeometry, Environment

        if dct["type"] != "APD":
            raise ValueError

        if dct["version"] != 1:
            raise ValueError

        properties = dct["properties"]
        geometry = APDGeometry.from_dict(properties["geometry"])
        environment = Environment.from_dict(properties["environment"])
        characteristics = APDCharacteristics.from_dict(properties["characteristics"])

        detector = cls(
            geometry=geometry,
            environment=environment,
            characteristics=characteristics,
        )

        data = dct["data"]

        if "photon" in data:
            detector.photon.array = np.asarray(data["photon"])

        scene = data.get("scene")  # type: Optional[Mapping]
        if scene is not None:
            detector.scene = Scene.from_dict(scene)

        if "pixel" in data:
            detector.pixel.array = np.asarray(data["pixel"])
        if "signal" in data:
            detector.signal.array = np.asarray(data["signal"])
        if "image" in data:
            detector.image.array = np.asarray(data["image"])
        if "charge" in data and data["charge"] is not None:
            charge_dct = data["charge"]
            detector.charge._array = np.asarray(charge_dct["array"])

            new_frame = charge_dct["frame"]  # type: pd.DataFrame
            previous_frame = detector.charge._frame  # type: pd.DataFrame
            detector.charge._frame = new_frame[previous_frame.columns]

        return detector

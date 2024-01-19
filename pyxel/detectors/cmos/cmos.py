#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

""":term:`CMOS` detector modeling class."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pyxel.data_structure import _get_array_if_initialized
from pyxel.detectors import Detector

if TYPE_CHECKING:
    import pandas as pd

    from pyxel.detectors import Characteristics, CMOSGeometry, Environment


class CMOS(Detector):
    """:term:`CMOS`-based detector class containing all detector attributes and data."""

    def __init__(
        self,
        geometry: "CMOSGeometry",
        environment: "Environment",
        characteristics: "Characteristics",
    ):
        self._geometry: CMOSGeometry = geometry
        self._characteristics: Characteristics = characteristics

        super().__init__(environment=environment)
        super()._initialize()

    def __eq__(self, other) -> bool:
        return (
            type(self) is type(other)
            and self.geometry == other.geometry
            and self.environment == other.environment
            and self.characteristics == other.characteristics
            and super().__eq__(other)
        )

    @property
    def geometry(self) -> "CMOSGeometry":
        """TBW."""
        return self._geometry

    @property
    def characteristics(self) -> "Characteristics":
        """TBW."""
        return self._characteristics

    # TODO: Refactor this
    def to_dict(self) -> Mapping:
        """Convert an instance of `CMOS` to a `dict`."""
        dct = {
            "version": 1,
            "type": "CMOS",
            "properties": {
                "geometry": self.geometry.to_dict(),
                "environment": self.environment.to_dict(),
                "characteristics": self.characteristics.to_dict(),
            },
            "data": {
                "photon": _get_array_if_initialized(self._photon),
                "photon_3d": (
                    None if self._photon3d is None else self._photon3d.to_dict()
                ),
                "pixel": _get_array_if_initialized(self._pixel),
                "signal": _get_array_if_initialized(self._signal),
                "image": _get_array_if_initialized(self._image),
                "data": None if self._data is None else self._data.to_dict(),
                "charge": (
                    None
                    if self._charge is None
                    else {
                        "array": self._charge.array.copy(),
                        "frame": self._charge.frame.copy(),
                    }
                ),
                "scene": (
                    None
                    if self._scene is None
                    else {
                        key.replace("/", "#"): value
                        for key, value in self._scene.to_dict().items()
                    }
                ),
            },
        }

        return dct

    # TODO: Refactor this
    @classmethod
    def from_dict(cls, dct: Mapping) -> "CMOS":
        """Create a new instance of `CMOS` from a `dict`."""
        # TODO: This is a simplistic implementation. Improve this.
        import numpy as np
        import xarray as xr
        from datatree import DataTree

        from pyxel.data_structure import Scene
        from pyxel.detectors import Characteristics, CMOSGeometry, Environment

        if dct["type"] != "CMOS":
            raise ValueError

        if dct["version"] != 1:
            raise ValueError

        properties = dct["properties"]
        geometry = CMOSGeometry.from_dict(properties["geometry"])
        environment = Environment.from_dict(properties["environment"])
        characteristics = Characteristics.from_dict(properties["characteristics"])

        detector = cls(
            geometry=geometry,
            environment=environment,
            characteristics=characteristics,
        )

        data: Mapping[str, Any] = dct["data"]

        detector.photon.update(data.get("photon"))
        if "photon_3d" in data and data["photon_3d"]:
            detector.photon3d.array = xr.DataArray(
                {
                    key.replace("#", "/"): value
                    for key, value in data["photon_3d"].items()
                }
            )
        detector.pixel.update(data.get("pixel"))
        detector.signal.update(data.get("signal"))
        detector.image.update(data.get("image"))

        if "data" in data:
            detector._data = DataTree.from_dict(
                {
                    key: xr.Dataset.from_dict(value)
                    for key, value in data["data"].items()
                }
            )
        if "scene" in data and (scene_dct := data["scene"]) is not None:
            detector.scene = Scene.from_dict(
                {key.replace("#", "/"): value for key, value in scene_dct.items()}
            )
        if "charge" in data and data["charge"] is not None:
            charge_dct = data["charge"]
            detector.charge._array = np.asarray(charge_dct["array"])

            new_frame: pd.DataFrame = charge_dct["frame"]
            previous_frame: pd.DataFrame = detector.charge._frame
            detector.charge._frame = new_frame[previous_frame.columns]

        return detector

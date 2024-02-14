#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import xarray as xr
from typing_extensions import Self

from pyxel.util import get_size


@dataclass
class WavelengthHandling:
    """Information about multi-wavelength."""

    cut_on: float
    cut_off: float
    resolution: int

    def __post_init__(self):
        assert 0 < self.cut_on <= self.cut_off
        assert self.resolution > 0

    def to_dict(self) -> dict:
        return {
            "cut_on": self.cut_on,
            "cut_off": self.cut_off,
            "resolution": self.resolution,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(
            cut_on=data["cut_on"],
            cut_off=data["cut_off"],
            resolution=data["resolution"],
        )

    def get_wavelengths(self) -> xr.DataArray:
        return xr.DataArray(
            np.arange(self.cut_on, self.cut_off, self.resolution),
            dims="wavelength",
            attrs={"units": "nm"},
        )


class Environment:
    """Environmental attributes of the detector.

    Parameters
    ----------
    temperature : float, optional
        Temperature of the detector. Unit: K
    wavelength : float, WavelengthHandling, optional
        Information about multi-wavelength. Unit: nm
    """

    def __init__(
        self,
        temperature: Optional[float] = None,
        wavelength: Union[float, WavelengthHandling, None] = None,
    ):
        if isinstance(temperature, (int, float)) and not (0.0 <= temperature <= 1000.0):
            raise ValueError("'temperature' must be between 0.0 and 1000.0.")

        if isinstance(wavelength, (int, float)) and not (wavelength > 0.0):
            raise ValueError("'wavelength' must be strictly positive.")

        if isinstance(wavelength, WavelengthHandling):
            if not (wavelength.cut_on < wavelength.cut_off):
                raise ValueError("'cut_on' must be strictly inferior to 'cut_off'.")
            if not (wavelength.resolution > 0):
                raise ValueError("'resolution' must be strictly positive and not 0.")

        self._temperature: Optional[float] = (
            float(temperature) if temperature is not None else None
        )

        self._wavelength: Union[float, WavelengthHandling, None] = (
            float(wavelength) if isinstance(wavelength, (int, float)) else wavelength
        )

        self._numbytes = 0

    def __repr__(self) -> str:
        cls_name: str = self.__class__.__name__
        return f"{cls_name}(temperature={self._temperature!r}, wavelength={self._wavelength!r})"

    def __eq__(self, other) -> bool:
        return (
            type(self) is type(other)
            and self._temperature == other._temperature
            and self._wavelength == other._wavelength
        )

    @property
    def temperature(self) -> float:
        """Get Temperature of the detector."""
        if self._temperature:
            return self._temperature
        else:
            raise ValueError("'temperature' not specified in detector environment.")

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set Temperature of the detector."""
        if not (0.0 <= value <= 1000.0):
            raise ValueError("'temperature' must be between 0.0 and 1000.0.")

        self._temperature = value

    @property
    def wavelength(self) -> Union[float, WavelengthHandling]:
        """Get wavelength of the detector."""
        if self._wavelength is None:
            raise ValueError("'wavelength' not specified in detector environment.")
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value: Union[float, WavelengthHandling]) -> None:
        """Set wavelength of the detector."""
        if isinstance(value, float) and not (value > 0.0):
            raise ValueError("'wavelength' must be strictly positive.")

        elif isinstance(value, WavelengthHandling):
            if not (value.cut_on < value.cut_off):
                raise ValueError("'cut_on' must be strictly inferior to 'cut_off'.")
            if not (value.resolution > 0):
                raise ValueError("'resolution' must be strictly positive and not 0.")
        else:
            raise ValueError("A WavelengthHandling object or a float must be provided.")

        self._wavelength = value

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

    def to_dict(self) -> Mapping:
        """Get the attributes of this instance as a `dict`."""
        if self._wavelength is None:
            wavelength_dict = {}
        elif isinstance(self._wavelength, (int, float)):
            wavelength_dict = {"wavelength": self._wavelength}
        else:
            wavelength_dict = self._wavelength.to_dict()
        return {"temperature": self._temperature} | wavelength_dict

    @classmethod
    def from_dict(cls, dct: Mapping) -> Self:
        """Create a new instance of `Geometry` from a `dict`."""

        value = dct.get("wavelength")

        if value is None:
            wavelength: Union[float, WavelengthHandling, None] = None
        elif isinstance(value, (int, float)):
            wavelength = float(value)
        elif isinstance(value, dict):
            wavelength = WavelengthHandling.from_dict(value)
        else:
            raise NotImplementedError

        return cls(temperature=dct.get("temperature"), wavelength=wavelength)

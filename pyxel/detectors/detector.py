#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Detector class."""
import collections
import typing as t
from math import sqrt
from pathlib import Path

import h5py as h5
import numpy as np

from pyxel import __version__
from pyxel.data_structure import Charge, Image, Photon, Pixel, Signal
from pyxel.detectors import Environment, Material
from pyxel.detectors.readout_properties import ReadoutProperties
from pyxel.util.memory import get_size, memory_usage_details

__all__ = ["Detector"]


# TODO: Add methods to save/load a `Detector` instance to the filesystem
#       Example of methods:
#           def to_fits(self, filename: Path):      # Save into one FITS file that contains multiple HDUs
#               ...
#           @classmethod
#           def from_fits(self, filename: Path) -> Detector     # Store into one FITS file
#               ...
#           def to_hdf5(...) / def from_hdf5(...)
#           def to_folder(...)  / def from_folder(...)
#           def to_yaml(...) ? / def from_yaml(...)
#           def to_asdf(...) ? / def from_asdf(...)
class Detector:
    """The detector class."""

    def __init__(self, material: Material, environment: Environment):
        self.material = material  # type: Material
        self.environment = environment  # type: Environment

        self.header = collections.OrderedDict()  # type: t.Dict[str, object]

        self._photon = None  # type: t.Optional[Photon]
        self._charge = None  # type: t.Optional[Charge]
        self._pixel = None  # type: t.Optional[Pixel]
        self._signal = None  # type: t.Optional[Signal]
        self._image = None  # type: t.Optional[Image]

        # This will be the memory of the detector where trapped charges will be saved
        self._memory = dict()  # type: t.Dict

        self.input_image = None  # type: t.Optional[np.ndarray]
        self._output_dir = None  # type: t.Optional[Path]  # TODO: Is it really needed ?

        self._readout_properties = None  # type: t.Optional["ReadoutProperties"]

        self._numbytes = get_size(self)

    @property
    def geometry(self):
        """TBW."""
        raise NotImplementedError

    @property
    def characteristics(self):
        """TBW."""
        raise NotImplementedError

    @property
    def has_photon(self) -> bool:
        """TBW."""
        return self._photon is not None

    @property
    def photon(self) -> Photon:
        """TBW."""
        if not self._photon:
            raise RuntimeError(
                "Photon array is not initialized ! "
                "Please use a 'Photon Generation' model"
            )
        return self._photon

    @photon.setter
    def photon(self, obj: Photon) -> None:
        self._photon = obj

    @property
    def charge(self) -> Charge:
        """TBW."""
        if not self._charge:
            raise RuntimeError("'charge' not initialized.")

        return self._charge

    @property
    def pixel(self) -> Pixel:
        """TBW."""
        if not self._pixel:
            raise RuntimeError("'pixel' not initialized.")

        return self._pixel

    @property
    def signal(self) -> Signal:
        """TBW."""
        if not self._signal:
            raise RuntimeError("'signal' not initialized.")

        return self._signal

    @property
    def image(self) -> Image:
        """TBW."""
        if not self._image:
            raise RuntimeError("'image' not initialized.")

        return self._image

    def reset(self, reset_all: bool = True) -> None:
        """TBW."""
        self._photon = None
        if reset_all:
            self._charge = Charge()
            self._pixel = Pixel(self.geometry)
            self._signal = Signal(self.geometry)
            self._image = Image(self.geometry)

    def empty(self, empty_all: bool = True) -> None:
        """Empty the data in the detector.

        Returns
        -------
        None
        """
        if self._photon:
            self.photon.array *= 0
        if empty_all:
            if self._charge:
                self.charge.nextid = 0
                self.charge.frame = self.charge.frame[0:0]
            if self._pixel:
                self.pixel.array *= 0
            if self._signal:
                self.signal.array *= 0
            if self._image:
                self.image.array *= 0

    # TODO: Set an `Output` object ?
    # TODO: Is it really needed ?
    def set_output_dir(self, path: t.Union[str, Path]) -> None:
        """Set output directory path."""
        self._output_dir = Path(path)

    # TODO: Get an `Output object ?
    # TODO: Is it really needed ?
    @property
    def output_dir(self) -> Path:
        """Output directory path."""
        if self._output_dir is None:
            raise RuntimeError("'output_dir' is not initialized.")

        return self._output_dir

    def set_readout(
        self,
        num_steps: int,
        start_time: float,
        end_time: float,
        ndreadout: bool = False,
        times_linear: bool = True,
    ) -> None:
        """Set sampling."""
        self._readout_properties = ReadoutProperties(
            num_steps=num_steps,
            start_time=start_time,
            end_time=end_time,
            ndreadout=ndreadout,
            times_linear=times_linear,
        )

    @property
    def time(self) -> float:
        """TBW."""
        if self._readout_properties is not None:
            return self._readout_properties.time
        else:
            raise ValueError("No sampling defined.")

    @time.setter
    def time(self, value: float) -> None:
        """TBW."""
        if self._readout_properties is not None:
            self._readout_properties.time = value
        else:
            raise ValueError("No sampling defined.")

    @property
    def start_time(self) -> float:
        """TBW."""
        if self._readout_properties is not None:
            return self._readout_properties.start_time
        else:
            raise ValueError("No sampling defined.")

    @start_time.setter
    def start_time(self, value: float) -> None:
        """TBW."""
        if self._readout_properties is not None:
            self._readout_properties.start_time = value
        else:
            raise ValueError("No sampling defined.")

    @property
    def absolute_time(self) -> float:
        """TBW."""
        if self._readout_properties is not None:
            return self._readout_properties.absolute_time
        else:
            raise ValueError("No sampling defined.")

    @property
    def time_step(self) -> float:
        """TBW."""
        if self._readout_properties is not None:
            return self._readout_properties.time_step
        else:
            raise ValueError("No sampling defined.")

    @time_step.setter
    def time_step(self, value: float) -> None:
        """TBW."""
        if self._readout_properties is not None:
            self._readout_properties.time_step = value
        else:
            raise ValueError("No sampling defined.")

    @property
    def times_linear(self) -> bool:
        """TBW."""
        if self._readout_properties is not None:
            return self._readout_properties.times_linear
        else:
            raise ValueError("No sampling defined.")

    @property
    def num_steps(self) -> int:
        """TBW."""
        if self._readout_properties is not None:
            return self._readout_properties.num_steps
        else:
            raise ValueError("No sampling defined.")

    @property
    def pipeline_count(self) -> float:
        """TBW."""
        if self._readout_properties is not None:
            return self._readout_properties.pipeline_count
        else:
            raise ValueError("No sampling defined.")

    @pipeline_count.setter
    def pipeline_count(self, value: int) -> None:
        """TBW."""
        if self._readout_properties is not None:
            self._readout_properties.pipeline_count = value
        else:
            raise ValueError("No sampling defined.")

    @property
    def read_out(self) -> bool:
        """TBW."""
        if self._readout_properties is not None:
            return self._readout_properties.read_out
        else:
            raise ValueError("No sampling defined.")

    @read_out.setter
    def read_out(self, value: bool) -> None:
        """TBW."""
        if self._readout_properties is not None:
            self._readout_properties.read_out = value
        else:
            raise ValueError("No sampling defined.")

    @property
    def is_dynamic(self) -> bool:
        """Return if detector is dynamic (time dependent) or not.

        By default it is not dynamic.
        """
        if self._readout_properties is not None:
            return True
        else:
            return False

    @property
    def non_destructive_readout(self) -> bool:
        """Return if detector readout mode is destructive or integrating.

        By default it is destructive (non-integrating).
        """
        if self._readout_properties is not None:
            return self._readout_properties.non_destructive
        else:
            raise ValueError("No sampling defined.")

    @property
    def e_thermal_velocity(self) -> float:
        """TBW.

        :return:
        """
        k_boltzmann = 1.38064852e-23  # J/K
        return sqrt(
            3
            * k_boltzmann
            * self.environment.temperature
            / self.material.e_effective_mass
        )

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

    def memory_usage(
        self, print_result: bool = True, human_readable: bool = True
    ) -> dict:
        """TBW.

        Returns
        -------
        dict
            Dictionary of attribute memory usage
        """
        attributes = [
            "_photon",
            "_charge",
            "_pixel",
            "_signal",
            "_image",
            "material",
            "environment",
            "_geometry",
            "_characteristics",
        ]

        return memory_usage_details(
            self, attributes, print_result=print_result, human_readable=human_readable
        )

    # TODO: Move this to another place
    def to_hdf5(self, filename: t.Union[str, Path]) -> None:
        """Convert the detector to a HDF5 object."""
        with h5.File(filename, "w") as h5file:
            h5file.attrs["type"] = self.__class__.__name__
            h5file.attrs["pyxel-version"] = str(__version__)
            detector_grp = h5file.create_group("detector")
            for array, name in zip(
                [
                    self.signal.array,
                    self.image.array,
                    self.photon.array,
                    self.pixel.array,
                    self.charge.frame,
                ],
                ["Signal", "Image", "Photon", "Pixel", "Charge"],
            ):
                dataset = detector_grp.create_dataset(name, shape=np.shape(array))
                dataset[:] = array

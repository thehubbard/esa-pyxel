"""Detector class."""
import collections
import typing as t
from math import sqrt
from pathlib import Path

import numpy as np

# from pyxel.detectors.characteristics import Characteristics
from pyxel.data_structure.charge import Charge
from pyxel.data_structure.image import Image
from pyxel.data_structure.photon import Photon
from pyxel.data_structure.pixel import Pixel
from pyxel.data_structure.signal import Signal
from pyxel.detectors.environment import Environment
# from pyxel.detectors.geometry import Geometry
from pyxel.detectors.material import Material

# from pyxel.detectors.cmos_geometry import CMOSGeometry                  # noqa: F401
# from pyxel.detectors.ccd_geometry import CCDGeometry                    # noqa: F401
# from pyxel.detectors.cmos_characteristics import CMOSCharacteristics    # noqa: F401
# from pyxel.detectors.ccd_characteristics import CCDCharacteristics      # noqa:


# FRED: There is a big flaw with this class.
#       A `Detector` instance can have a `CCDGeometry` and a `CMOSCharacteristics`.
#       This is not possible. We must solve this issue.
#       Note: Several solution are possible:
#         - Classes `Geometry`, `CCDGeometry` and `CMOSGeometry` have exactly the same attributes (not possible ?)
#         - Remove 'geometry' and 'characteristics' from this class. Implement them only in `CCD` and `CMOSDetector`
#         - Keep 'geometry' and 'chararacteristics' in this class. For class `CCD`,
#           add a new attribute 'extra_ccd_geometry',
#           for class `CMOSDetector`, add a new attribute 'extra_cmos_geometry'. Do the same for 'characteristics'
# FRED: Add methods to save/load a `Detector` instance to the filesystem
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

    def __init__(self,
                 material: Material,
                 environment: Environment):
        """TBW.

        :param material:
        :param environment:
        """
        self.material = material                    # type: Material
        self.environment = environment              # type: Environment

        self.header = collections.OrderedDict()     # type: t.Dict[str, object]

        self._photon = None  # type: t.Optional[Photon]
        self._charge = None  # type: t.Optional[Charge]
        self._pixel = None  # type: t.Optional[Pixel]
        self._signal = None  # type: t.Optional[Signal]
        self._image = None  # type: t.Optional[Image]

        self.input_image = None
        self._output_dir = None             # type: t.Optional[Path]

        self.start_time = 0.                # type: float
        self.end_time = 0.                  # type: float
        self.steps = 0                      # type: int
        self.time_step = 0.                 # type: float
        self._time = 0.                     # type: float
        self._dynamic = False               # type: bool
        self._non_destructive = False       # type: bool
        self.read_out = True                # type: bool
        self._all_time_steps_it = iter([])  # type: t.Iterator[float]

    @property
    def geometry(self):
        """TBW."""
        raise NotImplementedError

    @property
    def characteristics(self):
        """TBW."""
        raise NotImplementedError

    @property
    def photon(self) -> Photon:
        """TBW."""
        if not self._photon:
            raise RuntimeError("'photon' not initialized.")

        return self._photon

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

    # FRED: Rename to 'reset' ?
    def initialize(self, reset_all: bool = True) -> None:
        """TBW."""
        self._photon = Photon()
        if reset_all:
            self._charge = Charge()
            self._pixel = Pixel(self.geometry)
            self._signal = Signal(self.geometry)
            self._image = Image(self.geometry)

    def set_output_dir(self, path: t.Union[str, Path]) -> None:
        """Set output directory path."""
        self._output_dir = Path(path)

    @property
    def output_dir(self) -> Path:
        """Output directory path."""
        if self._output_dir is None:
            raise RuntimeError("'output_dir' is not initialized.")

        return self._output_dir

    def set_dynamic(self, time_step: float, steps: int, ndreadout: bool = False) -> None:
        """Switch on dynamic (time dependent) mode."""
        self._dynamic = True
        self.time_step = time_step
        self.steps = steps
        self._non_destructive = ndreadout
        self.end_time = self.time_step * self.steps

        all_time_steps = np.round(np.linspace(self.time_step, self.end_time, self.steps, endpoint=True), decimals=10)
        self._all_time_steps_it = map(float, all_time_steps)

    @property
    def is_dynamic(self) -> bool:
        """Return if detector is dynamic (time dependent) or not.

        By default it is not dynamic.
        """
        return self._dynamic

    @property
    def is_non_destructive_readout(self) -> bool:
        """Return if detector readout mode is destructive or integrating.

        By default it is destructive (non-integrating).
        """
        return self._non_destructive

    @property
    def e_thermal_velocity(self) -> float:
        """TBW.

        :return:
        """
        k_boltzmann = 1.38064852e-23  # J/K
        return sqrt(3 * k_boltzmann * self.environment.temperature / self.material.e_effective_mass)

    @property
    def time(self) -> float:     # TODO
        """TBW."""
        return self._time

    # FRED: This method is used in 'run.py'. We could implement this as an iterator.
    def elapse_time(self) -> float:
        """TBW."""
        try:
            self._time = float(next(self._all_time_steps_it))
        except StopIteration:
            self._time = 0.0
        return self._time

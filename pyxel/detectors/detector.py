"""Detector class."""
from math import sqrt
import collections
import typing as t
import numpy as np
from pathlib import Path

# from astropy import units as u
from pyxel.detectors import Material
from pyxel.detectors import Environment
from pyxel.data_structure.charge import Charge
from pyxel.data_structure.photon import Photon
from pyxel.data_structure.pixel import Pixel
from pyxel.data_structure.signal import Signal
from pyxel.data_structure.image import Image
# from pyxel.detectors.cmos_geometry import CMOSGeometry
# from pyxel.detectors.ccd_geometry import CCDGeometry
# from pyxel.detectors.cmos_characteristics import CMOSCharacteristics
# from pyxel.detectors.ccd_characteristics import CCDCharacteristics
import esapy_config.config as ec
import attr


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
@ec.config
class Detector:
    """The detector class."""

    material = ec.setting(type=Material)
    environment = ec.setting(type=Environment)

    # FRED: Move these in `Geometry` ??
    photon = ec.setting(type=t.Optional[Photon],
                        default=attr.Factory(lambda self: Photon(self.geometry), takes_self=True), init=False)
    charge = ec.setting(type=t.Optional[Charge], factory=Charge, init=False)
    pixel = ec.setting(type=t.Optional[Pixel], default=attr.Factory(lambda self: Pixel(self.geometry), takes_self=True),
                       init=False)
    signal = ec.setting(type=t.Optional[Signal],
                        default=attr.Factory(lambda self: Signal(self.geometry), takes_self=True), init=False)
    image = ec.setting(type=t.Optional[Image], default=attr.Factory(lambda self: Image(self.geometry), takes_self=True),
                       init=False)

    # FRED: Internal attributes (==> 'init=False')
    header = ec.setting(type=collections.OrderedDict, factory=collections.OrderedDict, init=False)
    input_image = ec.setting(default=None, init=False)
    output_dir = ec.setting(type=Path, default=Path(), init=False)
    start_time = ec.setting(type=float, default=0., init=False)
    end_time = ec.setting(type=float, default=0., init=False)
    steps = ec.setting(type=int, default=0, init=False)
    time_step = ec.setting(type=float, default=0., init=False)
    time = ec.setting(type=float, default=0., init=False)
    dynamic = ec.setting(type=bool, default=False, init=False)
    non_destructive = ec.setting(type=bool, default=False, init=False)
    read_out = ec.setting(type=bool, default=True, init=False)
    all_time_steps = ec.setting(default=None, init=False)

    # def __init__(self,
    #              geometry: Geometry,
    #              material: Material,
    #              environment: Environment,
    #              characteristics: Characteristics,
    #              photon: Photon = None,
    #              charge: Charge = None,
    #              pixel: Pixel = None,
    #              signal: Signal = None,
    #              image: Image = None) -> None:
    #     """TBW.
    #
    #     :param geometry:
    #     :param material:
    #     :param environment:
    #     :param characteristics:
    #     :param photon:
    #     :param charge:
    #     :param pixel:
    #     :param signal:
    #     :param image:
    #     """
    #     self._geometry = geometry                   # type: Geometry
    #     self._characteristics = characteristics     # type: Characteristics
    #     self.material = material                    # type: Material
    #     self.environment = environment              # type: Environment
    #     self.header = collections.OrderedDict()     # type: t.Dict[str, object]
    #
    #     if photon:
    #         self.photon = photon
    #     else:
    #         # FRED: This could be assigned directly in the 'default' field of `attr` or `esapy_config`
    #         self.photon = Photon(self.geometry)
    #     if charge:
    #         self.charge = charge
    #     else:
    #         # FRED: This could be assigned directly in the 'default' field of `attr` or `esapy_config`
    #         self.charge = Charge()
    #     if pixel:
    #         self.pixel = pixel
    #     else:
    #         # FRED: This could be assigned directly in the 'default' field of `attr` or `esapy_config`
    #         self.pixel = Pixel(self.geometry)
    #     if signal:
    #         self.signal = signal
    #     else:
    #         # FRED: This could be assigned directly in the 'default' field of `attr` or `esapy_config`
    #         self.signal = Signal(self.geometry)
    #     if image:
    #         self.image = image
    #     else:
    #         # FRED: This could be assigned directly in the 'default' field of `attr` or `esapy_config`
    #         self.image = Image(self.geometry)
    #
    #     self.input_image = None
    #     self._output_dir = None                         # type: t.Optional[str]
    #
    #     self.start_time = 0.                            # type: float
    #     self.end_time = 0.                              # type: float
    #     self.steps = 0                                  # type: int
    #     self.time_step = 0.                             # type: float
    #     self._time = 0.                                 # type: float
    #     self._dynamic = False                           # type: bool
    #     self._non_destructive = False                   # type: bool
    #     self.read_out = True                            # type: bool
    #     self._all_time_steps = None

    def __attrs_post_init__(self):
        """TBW."""
        self.initialize()

    @property
    def geometry(self):
        """TBW."""
        raise NotImplementedError

    @property
    def characteristics(self):
        """TBW."""
        raise NotImplementedError

    # FRED: Rename to 'reset' ?
    def initialize(self, reset_all: bool = True) -> None:
        """TBW."""
        self.photon = Photon()             # type: Photon
        if reset_all:
            self.charge = Charge()                      # type: Charge
            self.pixel = Pixel(self.geometry)           # type: Pixel
            self.signal = Signal(self.geometry)         # type: Signal
            self.image = Image(self.geometry)           # type: Image

    def set_output_dir(self, path: t.Optional[t.Union[str, Path]] = None) -> None:
        """Set output directory path."""
        self.output_dir = Path(path)

    # @property
    # def output_dir(self):
    #     """Output directory path."""
    #     return self._output_dir

    def set_dynamic(self, time_step: float, steps: int, ndreadout: bool = False) -> None:
        """Switch on dynamic (time dependent) mode."""
        self.dynamic = True
        self.time_step = time_step
        self.steps = steps
        self.non_destructive = ndreadout
        self.end_time = self.time_step * self.steps
        self.all_time_steps = np.nditer(np.round(np.linspace(self.time_step, self.end_time,
                                                             self.steps, endpoint=True), decimals=10))

    @property
    def is_dynamic(self) -> bool:
        """Return if detector is dynamic (time dependent) or not.

        By default it is not dynamic.
        """
        return self.dynamic

    @property
    def is_non_destructive_readout(self) -> bool:
        """Return if detector readout mode is destructive or integrating.

        By default it is destructive (non-integrating).
        """
        return self.non_destructive

    # FRED: This should be solved
    # @property
    # def geometry(self):
    #     """TBW."""
    #     if isinstance(self._geometry, CMOSGeometry):
    #         return t.cast(CMOSGeometry, self._geometry)
    #     elif isinstance(self._geometry, CCDGeometry):
    #         return t.cast(CCDGeometry, self._geometry)
    #     elif isinstance(self._geometry, Geometry):
    #         return t.cast(Geometry, self._geometry)
    #
    # # FRED: This should be solved
    # @property
    # def characteristics(self):
    #     """TBW."""
    #     if isinstance(self._characteristics, CMOSCharacteristics):
    #         return t.cast(CMOSCharacteristics, self._characteristics)
    #     elif isinstance(self._characteristics, CCDCharacteristics):
    #         return t.cast(CCDCharacteristics, self._characteristics)
    #     elif isinstance(self._characteristics, Characteristics):
    #         return t.cast(Characteristics, self._characteristics)

    @property
    def e_thermal_velocity(self) -> float:
        """TBW.

        :return:
        """
        k_boltzmann = 1.38064852e-23  # J/K
        return sqrt(3 * k_boltzmann * self.environment.temperature / self.material.e_effective_mass)

    # @property
    # def time(self) -> float:     # TODO
    #     """TBW."""
    #     return self._time

    # FRED: This method is used in 'run.py'. We could implement this as an iterator.
    def elapse_time(self) -> float:
        """TBW."""
        try:
            self.time = float(next(self.all_time_steps))
        except StopIteration:
            self.time = None
        return self.time

"""CCD detector modeling class."""
from math import sqrt
import collections
import typing as t  # noqa: F401

# from astropy import units as u
from pyxel.detectors.geometry import Geometry
from pyxel.detectors.material import Material
from pyxel.detectors.environment import Environment
from pyxel.detectors.characteristics import Characteristics
from pyxel.data_structure.charge import Charge                  # noqa: F401
from pyxel.data_structure.photon import Photon                  # noqa: F401
from pyxel.data_structure.pixel import Pixel                    # noqa: F401
from pyxel.data_structure.signal import Signal                  # noqa: F401
from pyxel.data_structure.image import Image                    # noqa: F401
from pyxel.detectors.cmos_geometry import CMOSGeometry          # noqa: F401
from pyxel.detectors.ccd_geometry import CCDGeometry            # noqa: F401


class Detector:
    """The CCD detector class."""

    def __init__(self,
                 geometry: Geometry,
                 material: Material,
                 environment: Environment,
                 characteristics: Characteristics,
                 photon: Photon = None,
                 charge: Charge = None,
                 pixel: Pixel = None,
                 signal: Signal = None,
                 image: Image = None) -> None:
        """TBW.

        :param geometry:
        :param material:
        :param environment:
        :param characteristics:
        :param photon:
        :param charge:
        :param pixel:
        :param signal:
        :param image:
        """
        self.geometry = geometry                   # type: Geometry
        self.material = material                   # type: Material
        self.environment = environment             # type: Environment
        self.characteristics = characteristics     # type: Characteristics
        self.header = collections.OrderedDict()    # type: t.Dict[str, object]

        self.photon = Photon(self.geometry)        # type: Photon
        self.charge = Charge()                     # type: Charge
        self.pixel = Pixel(self.geometry)          # type: Pixel
        self.signal = Signal(self.geometry)        # type: Signal
        self.image = Image(self.geometry)          # type: Image

        if photon:
            self.photon = photon
        if charge:
            self.charge = charge
        if pixel:
            self.pixel = pixel
        if signal:
            self.signal = signal
        if image:
            self.image = image

        self.input_image = None

    def __getstate__(self):
        """TBW.

        :return:
        """
        return {
            'geometry': self.geometry,
            'material': self.material,
            'environment': self.environment,
            'characteristics': self.characteristics,
            'photon': self.photon,
            'charge': self.charge,
            'pixel': self.pixel,
            'signal': self.signal,
            'image': self.image,
            'input_image': self.input_image
        }

    @property
    def e_thermal_velocity(self):
        """TBW.

        :return:
        """
        k_boltzmann = 1.38064852e-23  # J/K
        return sqrt(3 * k_boltzmann * self.environment.temperature / self.material.e_effective_mass)

    def get_geometry(self):
        """TBW."""
        if isinstance(self.geometry, CMOSGeometry):
            return t.cast(CMOSGeometry, self.geometry)
        elif isinstance(self.geometry, CCDGeometry):
            return t.cast(CCDGeometry, self.geometry)
        elif isinstance(self.geometry, Geometry):
            return t.cast(Geometry, self.geometry)

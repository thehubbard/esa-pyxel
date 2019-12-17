"""CCD detector modeling class."""

from pyxel.data_structure.charge import Charge  # noqa: F401
from pyxel.data_structure.image import Image  # noqa: F401
from pyxel.data_structure.photon import Photon  # noqa: F401
from pyxel.data_structure.pixel import Pixel  # noqa: F401
from pyxel.data_structure.signal import Signal  # noqa: F401
from pyxel.detectors.ccd_characteristics import CCDCharacteristics
from pyxel.detectors.ccd_geometry import CCDGeometry
from pyxel.detectors.detector import Detector
from pyxel.detectors.environment import Environment
from pyxel.detectors.material import Material


class CCD(Detector):
    """Charge-Coupled Device class containing all detector attributes and data."""

    def __init__(
        self,
        geometry: CCDGeometry,
        material: Material,
        environment: Environment,
        characteristics: CCDCharacteristics,
    ):
        """TBW.

        :param geometry:
        :param material:
        :param environment:
        :param characteristics:
        """
        self._geometry = geometry  # type: CCDGeometry
        self._characteristics = characteristics  # type: CCDCharacteristics

        super().__init__(material=material, environment=environment)
        super().initialize()

    @property
    def geometry(self) -> CCDGeometry:
        """TBW."""
        return self._geometry

    @property
    def characteristics(self) -> CCDCharacteristics:
        """TBW."""
        return self._characteristics

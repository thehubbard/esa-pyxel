#   --------------------------------------------------------------------------
#   Copyright 2018 SCI-FIV, ESA (European Space Agency)
#   --------------------------------------------------------------------------
"""CCD detector modeling class."""

from pyxel.detectors.detector import Detector
from pyxel.detectors.ccd_geometry import CCDGeometry
from pyxel.detectors.material import Material
from pyxel.detectors.environment import Environment
from pyxel.detectors.ccd_characteristics import CCDCharacteristics
from pyxel.data_structure.charge import Charge  # noqa: F401
from pyxel.data_structure.photon import Photon  # noqa: F401
from pyxel.data_structure.pixel import Pixel    # noqa: F401
from pyxel.data_structure.signal import Signal  # noqa: F401
from pyxel.data_structure.image import Image    # noqa: F401


# FRED: This class could be implement with ESAPY_CONFIG
class CCD(Detector):
    """Charge-Coupled Device class containing all detector attributes and data."""

    def __init__(self,
                 geometry: CCDGeometry,
                 material: Material,
                 environment: Environment,
                 characteristics: CCDCharacteristics,
                 photon: Photon = None,         # FRED: Use t.Optional[Photon]
                 charge: Charge = None,         # FRED: Use t.Optional
                 pixel: Pixel = None,           # FRED: Use t.Optional
                 signal: Signal = None,         # FRED: Use t.Optional
                 image: Image = None) -> None:  # FRED: Use t.Optional + -> None is not needed
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
        super().__init__(geometry=geometry,
                         material=material,
                         environment=environment,
                         characteristics=characteristics,
                         photon=photon,
                         charge=charge,
                         pixel=pixel,
                         signal=signal,
                         image=image)

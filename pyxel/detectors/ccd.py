#   --------------------------------------------------------------------------
#   Copyright 2019 SCI-FIV, ESA (European Space Agency)
#   --------------------------------------------------------------------------
# HANS: remove copyright
"""CCD detector modeling class."""

import typing as t
from pyxel.detectors.detector import Detector
from pyxel.detectors.ccd_geometry import CCDGeometry
from pyxel.detectors.material import Material
from pyxel.detectors.environment import Environment
from pyxel.detectors.ccd_characteristics import CCDCharacteristics
from pyxel.data_structure.charge import Charge
from pyxel.data_structure.photon import Photon
from pyxel.data_structure.pixel import Pixel
from pyxel.data_structure.signal import Signal
from pyxel.data_structure.image import Image


class CCD(Detector):
    """Charge-Coupled Device class containing all detector attributes and data."""

    def __init__(self,
                 geometry: CCDGeometry,
                 material: Material,
                 environment: Environment,
                 characteristics: CCDCharacteristics,
                 photon: t.Optional[Photon] = None,
                 charge: t.Optional[Charge] = None,
                 pixel: t.Optional[Pixel] = None,
                 signal: t.Optional[Signal] = None,
                 image: t.Optional[Image] = None):
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

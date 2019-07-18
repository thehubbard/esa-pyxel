#   --------------------------------------------------------------------------
#   Copyright 2019 SCI-FIV, ESA (European Space Agency)
#   --------------------------------------------------------------------------
# HANS: remove copyright
"""CCD detector modeling class."""

# import typing as t
from pyxel.detectors.ccd_geometry import CCDGeometry
from pyxel.detectors.environment import Environment
from pyxel.detectors.ccd_characteristics import CCDCharacteristics
# from pyxel.data_structure.charge import Charge
# from pyxel.data_structure.photon import Photon
# from pyxel.data_structure.pixel import Pixel
# from pyxel.data_structure.signal import Signal
# from pyxel.data_structure.image import Image
from pyxel.detectors.detector import Detector
from pyxel.detectors.material import Material


# if t.TYPE_CHECKING:
#     from pyxel.detectors.material import Material


class CCD(Detector):
    """Charge-Coupled Device class containing all detector attributes and data."""

    def __init__(self,
                 geometry: CCDGeometry,
                 material: "Material",
                 environment: Environment,
                 characteristics: CCDCharacteristics,
                 # photon: t.Optional[Photon] = None,
                 # charge: t.Optional[Charge] = None,
                 # pixel: t.Optional[Pixel] = None,
                 # signal: t.Optional[Signal] = None,
                 # image: t.Optional[Image] = None
                 ):
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
        self._geometry = geometry  # type: CCDGeometry
        self._characteristics = characteristics  # type: CCDCharacteristics

        super().__init__(material=material,
                         environment=environment,
                         # photon=photon,
                         # charge=charge,
                         # pixel=pixel,
                         # signal=signal,
                         # image=image
                         )

    @property
    def geometry(self) -> CCDGeometry:
        """TBW."""
        return self._geometry

    @property
    def characteristics(self) -> CCDCharacteristics:
        """TBW."""
        return self._characteristics

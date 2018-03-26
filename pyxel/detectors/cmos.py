#   --------------------------------------------------------------------------
#   Copyright 2018 SCI-FIV, ESA (European Space Agency)
#   --------------------------------------------------------------------------
"""CMOS detector modeling class."""

from pyxel.detectors.detector import Detector
from pyxel.detectors.environment import Environment
from pyxel.detectors.cmos_geometry import CMOSGeometry
from pyxel.detectors.cmos_characteristics import CMOSCharacteristics


class CMOS(Detector):
    """TBW."""

    def __init__(self,
                 geometry: CMOSGeometry,
                 environment: Environment,
                 characteristics: CMOSCharacteristics) -> None:
        """TBW.

        :param geometry:
        :param environment:
        :param characteristics:
        """
        super().__init__(geometry=geometry,
                         environment=environment,
                         characteristics=characteristics)

    def copy(self):
        """TBW."""
        cpy = super().copy()
        kwargs = {
            'geometry': cpy.geometry,
            'environment': cpy.environment,
            'characteristics': cpy.characteristics,
        }
        return CMOS(**kwargs)
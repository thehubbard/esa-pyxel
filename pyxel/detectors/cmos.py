#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""CMOS detector modeling class."""

from pyxel.detectors.cmos_characteristics import CMOSCharacteristics
from pyxel.detectors.cmos_geometry import CMOSGeometry
from pyxel.detectors.detector import Detector
from pyxel.detectors.environment import Environment
from pyxel.detectors.material import Material

# from pyxel.data_structure.charge import Charge
# from pyxel.data_structure.photon import Photon
# from pyxel.data_structure.pixel import Pixel
# from pyxel.data_structure.signal import Signal
# from pyxel.data_structure.image import Image


class CMOS(Detector):
    """CMOS-based detector class containing all detector attributes and data."""

    def __init__(
        self,
        geometry: CMOSGeometry,
        material: Material,
        environment: Environment,
        characteristics: CMOSCharacteristics,
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
        """
        self._geometry = geometry  # type: CMOSGeometry
        self._characteristics = characteristics  # type: CMOSCharacteristics

        super().__init__(material=material, environment=environment)
        super().initialize()

    @property
    def geometry(self) -> CMOSGeometry:
        """TBW."""
        return self._geometry

    @property
    def characteristics(self) -> CMOSCharacteristics:
        """TBW."""
        return self._characteristics

"""The detector attribute and data structure subpackage."""

# Warning: Import order matters
from pyxel.detectors.material import Material
from pyxel.detectors.environment import Environment
from pyxel.detectors.ccd import CCD
from pyxel.detectors.cmos import CMOS
from pyxel.detectors.geometry import Geometry
from pyxel.detectors.characteristics import Characteristics
from pyxel.detectors.detector import Detector
from pyxel.detectors.optics import Optics


# TODO: Is '__all__' really necessary ?
__all__ = ['Detector', 'CCD', 'CMOS',
           'Geometry', 'Characteristics', 'Material', 'Environment', 'Optics']

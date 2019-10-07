"""The detector attribute and data structure subpackage."""

# flake8: noqa
# Warning: Import order matters
from .material import Material
from .environment import Environment
from .ccd import CCD
from .cmos import CMOS
from .geometry import Geometry
from .characteristics import Characteristics
from .detector import Detector
from .optics import Optics


# TODO: Is '__all__' really necessary ?
# __all__ = ['Detector', 'CCD', 'CMOS',
#            'Geometry', 'Characteristics', 'Material', 'Environment', 'Optics']

"""PyXEL is an detector simulation framework."""


from pyxel.io.yaml_processor import load_config
from pyxel.pipelines import ccd_pipeline
from pyxel.pipelines import cmos_pipeline
from pyxel.pipelines import models
from pyxel.pipelines import processor
from pyxel.pipelines.model_registry import validate
from pyxel.pipelines.model_registry import argument
from pyxel.pipelines.model_registry import register
from pyxel.pipelines.model_registry import registry
from pyxel.pipelines.model_registry import parameters
from pyxel.pipelines.model_registry import ValidationError
from pyxel.pipelines.model_registry import MetaModel
from pyxel.pipelines.model_group import ModelFunction
from pyxel.pipelines.model_group import ModelGroup
from pyxel.pipelines.detector_pipeline import DetectionPipeline
from pyxel.pipelines.processor import Processor


__all__ = ['load_config', 'ccd_pipeline', 'cmos_pipeline', 'models', 'processor',
           'validate', 'argument', 'register', 'registry', 'parameters', 'ValidationError',
           'MetaModel', 'ModelFunction', 'ModelGroup',
           'DetectionPipeline', 'Processor']

__appname__ = 'Pyxel'
__author__ = 'David Lucsanyi'
__author_email__ = 'david.lucsanyi@esa.int'
__pkgname__ = 'pyxel'

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

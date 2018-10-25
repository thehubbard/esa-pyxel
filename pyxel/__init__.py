"""PyXEL is an detector simulation framework."""

import esapy_config as om

from pyxel.pipelines import processor
from pyxel.pipelines.model_group import ModelFunction
from pyxel.pipelines.model_group import ModelGroup
from pyxel.pipelines.detector_pipeline import DetectionPipeline
from pyxel.pipelines.processor import Processor
from pyxel.detectors.detector import Detector


__all__ = ['models', 'processor',
           'ModelFunction', 'ModelGroup',
           'DetectionPipeline', 'Processor', 'Detector']

__appname__ = 'Pyxel'
__author__ = 'David Lucsanyi'
__author_email__ = 'david.lucsanyi@esa.int'
__pkgname__ = 'pyxel'

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


def register(group, maybe_func=None, **kwargs):
    """TBW.

    :param group:
    :param maybe_func:
    :param kwargs:
    :return:
    """
    enabled = kwargs.pop('enabled', True)
    ignore_args = kwargs.pop('ignore_args', ['detector'])
    name = kwargs.pop('name', None)
    metadata = kwargs
    metadata['group'] = group
    return om.register(maybe_func, ignore_args, name, enabled, metadata)


def define_pyxel_loader():
    """TBW."""
    from pyxel.pipelines.parametric import StepValues
    from pyxel.pipelines.parametric import Configuration
    from pyxel.pipelines.model_group import ModelFunction
    from pyxel.pipelines.model_group import ModelGroup

    om.ObjectModelLoader.add_class_ref(['processor', 'class'])
    om.ObjectModelLoader.add_class_ref(['processor', 'detector', 'class'])
    om.ObjectModelLoader.add_class_ref(['processor', 'detector', None, 'class'])
    om.ObjectModelLoader.add_class_ref(['processor', 'pipeline', 'class'])

    om.ObjectModelLoader.add_class(Configuration, ['simulation'])
    om.ObjectModelLoader.add_class(StepValues, ['simulation', 'steps'], is_list=True)
    om.ObjectModelLoader.add_class(ModelGroup, ['processor', 'pipeline', None])
    om.ObjectModelLoader.add_class(ModelFunction, ['processor', 'pipeline', None, None])


define_pyxel_loader()

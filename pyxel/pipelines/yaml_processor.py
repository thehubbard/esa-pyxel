import functools
from pathlib import Path

import numpy as np
import yaml

from pyxel.detectors.ccd import CCD
from pyxel.detectors.cmos import CMOS
import pyxel.detectors.ccd_characteristics
import pyxel.detectors.cmos_characteristics
import pyxel.detectors.geometry
import pyxel.detectors.cmos_geometry
import pyxel.detectors.environment
from pyxel.pipelines import detection_pipeline
from pyxel.util import fitsfile
from pyxel.util import util


class PyxelLoader(yaml.SafeLoader):
    pass


def _constructor_processor(loader: PyxelLoader, node: yaml.MappingNode):
    mapping = loader.construct_mapping(node, deep=True)     # type: dict

    obj = detection_pipeline.Processor(**mapping)

    return obj


def _constructor_ccd_pipeline(loader: PyxelLoader, node: yaml.MappingNode):
    mapping = loader.construct_mapping(node, deep=True)     # type: dict

    obj = detection_pipeline.CCDDetectionPipeline(**mapping)

    return obj


def _constructor_cmos_pipeline(loader: PyxelLoader, node: yaml.MappingNode):
    mapping = loader.construct_mapping(node, deep=True)     # type: dict

    obj = detection_pipeline.CMOSDetectionPipeline(**mapping)

    return obj


def _constructor_ccd(loader: PyxelLoader, node: yaml.MappingNode):
    mapping = loader.construct_mapping(node, deep=True)  # type: dict

    if 'geometry' in mapping:
        geometry = pyxel.detectors.geometry.Geometry(**mapping['geometry'])
    else:
        geometry = None

    if 'environment' in mapping:
        environment = pyxel.detectors.environment.Environment(**mapping['environment'])
    else:
        environment = None

    if 'characteristics' in mapping:
        characteristics = pyxel.detectors.ccd_characteristics.CCDCharacteristics(**mapping['characteristics'])
    else:
        characteristics = None

    photons = mapping.get('photons', None)
    image = mapping.get('image', None)

    obj = CCD(photons=photons,
              image=image,
              geometry=geometry,
              environment=environment,
              characteristics=characteristics)

    return obj


def _constructor_cmos(loader: PyxelLoader, node: yaml.MappingNode):
    mapping = loader.construct_mapping(node, deep=True)  # type: dict

    if 'geometry' in mapping:
        geometry = pyxel.detectors.cmos_geometry.CMOSGeometry(**mapping['geometry'])
    else:
        geometry = None

    if 'environment' in mapping:
        environment = pyxel.detectors.environment.Environment(**mapping['environment'])
    else:
        environment = None

    if 'characteristics' in mapping:
        characteristics = pyxel.detectors.cmos_characteristics.CMOSCharacteristics(**mapping['characteristics'])
    else:
        characteristics = None

    photons = mapping.get('photons', None)
    image = mapping.get('image', None)

    obj = CMOS(photons=photons,
               image=image,
               geometry=geometry,
               environment=environment,
               characteristics=characteristics)

    return obj


def _constructor_from_file(loader: PyxelLoader, node: yaml.ScalarNode):
    filename = Path(loader.construct_scalar(node))
    if filename.suffix.lower().startswith('.fit'):
        result = fitsfile.FitsFile(str(filename)).data
    else:
        result = np.fromfile(str(filename), dtype=float, sep=' ')
    return result


def _constructor_models(loader: PyxelLoader, node: yaml.ScalarNode):

    if isinstance(node, yaml.ScalarNode):
        # OK: no kwargs provided
        mapping = {}  # type: dict
    else:
        mapping = loader.construct_mapping(node, deep=True)

    return detection_pipeline.Models(mapping)


def _constructor_function(loader: PyxelLoader, node: yaml.ScalarNode):
    mapping = loader.construct_mapping(node)             # type: dict

    function_name = mapping['name']                      # type: str
    kwargs = mapping.get('kwargs', {})
    args = mapping.get('args', {})

    func = util.evaluate_reference(function_name)

    return functools.partial(func, *args, **kwargs)


PyxelLoader.add_constructor('!PROCESSOR', _constructor_processor)

PyxelLoader.add_constructor('!CCD_PIPELINE', _constructor_ccd_pipeline)
PyxelLoader.add_constructor('!CMOS_PIPELINE', _constructor_cmos_pipeline)

PyxelLoader.add_constructor('!CCD', _constructor_ccd)
PyxelLoader.add_constructor('!CMOS', _constructor_cmos)

PyxelLoader.add_constructor('!from_file', _constructor_from_file)
PyxelLoader.add_constructor('!function', _constructor_function)
PyxelLoader.add_constructor('!models', _constructor_models)


def load_config(yaml_file):

    with open(yaml_file, 'r') as file_obj:
        cfg = yaml.load(file_obj, Loader=PyxelLoader)

    return cfg

"""TBW."""
import typing as t
import yaml
from pathlib import Path

try:
    # Use LibYAML library
    from yaml import CSafeLoader as SafeLoader  # type: ignore
except ImportError:
    from yaml import SafeLoader  # noqa: F401

from pyxel.util import objmod as om
from pyxel.pipelines.parametric import StepValues
from pyxel.pipelines.parametric import ParametricConfig
from pyxel.pipelines.model_group import ModelFunction
from pyxel.pipelines.model_group import ModelGroup


class PyxelLoader(yaml.SafeLoader):
    """Custom `SafeLoader` that constructs Pyxel objects.

    This class is not directly instantiated by user code, but instead is
    used to maintain the available constructor functions that are
    called when parsing a YAML stream.  See the `PyYaml documentation
    <http://pyyaml.org/wiki/PyYAMLDocumentation>`_ for details of the
    class signature.
    """

    class_paths = []  # type: t.List[t.List[str]]

    @classmethod
    def add_class(cls,
                  klass: type,
                  paths: list,
                  is_list: bool=False):
        """TBW.

        :param klass:
        :param paths:
        :param is_list:
        :return:
        """
        ClassConstructor(cls, klass, paths, is_list)

    @classmethod
    def add_class_ref(cls, path_to_class: list):
        """TBW.

        :param path_to_class:
        :return:
        """
        if path_to_class not in cls.class_paths:
            cls.class_paths.append(path_to_class)

        path_to_obj = path_to_class[:-1]

        cls.add_path_resolver('!Class', path_to_class)
        cls.add_path_resolver('!Object', path_to_obj)

        cls.add_constructor('!Class', _constructor_class)
        cls.add_constructor('!Object', _constructor_object)


class PyxelDumper(yaml.SafeDumper):
    """Custom `SafeDumper` that represents Pyxel objects.

    This class is not directly instantiated by user code, but instead is
    used to maintain the available constructor functions that are
    called when parsing a YAML stream.  See the `PyYaml documentation
    <http://pyyaml.org/wiki/PyYAMLDocumentation>`_ for details of the
    class signature.
    """


def _constructor_object(loader: PyxelLoader, node: yaml.MappingNode):
    """TBW.

    :param loader:
    :param node:
    :return:
    """
    if isinstance(node, yaml.ScalarNode):
        result = node.value
    elif isinstance(node, yaml.SequenceNode):
        result = loader.construct_sequence(node, deep=True)
    else:
        result = loader.construct_mapping(node, deep=True)
        if 'class' in result:
            class_name = result.pop('class')
            try:
                klass = om.evaluate_reference(class_name)
                result = klass(**result)
            except TypeError as exc:
                print('Cannot evaluate class: %s' % str(exc))
                return
    return result


def _constructor_class(loader: PyxelLoader, node: yaml.ScalarNode):
    """TBW.

    :param loader:
    :param node:
    :return:
    """
    return node.value


class ClassConstructor:
    """TBW."""

    def __init__(self,
                 loader: t.Type[yaml.SafeLoader],
                 klass: type,
                 paths: t.List[str],
                 is_list: bool=False) -> None:
        """TBW.

        :param klass:
        :param paths:
        :param is_list:
        """
        loader.add_path_resolver('!%s' % klass.__name__, paths)
        loader.add_constructor('!%s' % klass.__name__, self.__call__)
        self.klass = klass
        self.paths = paths
        self.is_list = is_list

    def __call__(self, loader, node):
        """TBW.

        :param loader:
        :param node:
        :return:
        """
        if isinstance(node, yaml.ScalarNode):
            obj = node.value

        elif isinstance(node, yaml.SequenceNode):
            coll = loader.construct_sequence(node, deep=True)
            if self.is_list:
                obj = [self.klass(**kwargs) for kwargs in coll]
            else:
                obj = self.klass(coll)

        elif isinstance(node, yaml.MappingNode):
            coll = loader.construct_mapping(node, deep=True)
            obj = self.klass(**coll)

        else:
            raise RuntimeError('Invalid node: %r' % node)

        return obj


# PyxelLoader.add_path_resolver('!Class', ['processor', 'class'])
# PyxelLoader.add_path_resolver('!Object', ['processor'])
#
# PyxelLoader.add_path_resolver('!Class', ['processor', 'detector', None, 'class'])
# PyxelLoader.add_path_resolver('!Object', ['processor', 'detector', None])
#
# PyxelLoader.add_path_resolver('!Class', ['processor', 'detector', 'class'])
# PyxelLoader.add_path_resolver('!Object', ['processor', 'detector'])
#
# # NOTE 1: not called - refer to NOTE 2 below
# PyxelLoader.add_path_resolver('!Class', ['processor', 'pipeline', 'class'])
# PyxelLoader.add_path_resolver('!Object', ['processor', 'pipeline'])
# PyxelLoader.add_path_resolver('!Object', ['processor', 'pipeline', None])
#
# PyxelLoader.add_constructor('!Class', _constructor_class)
# PyxelLoader.add_constructor('!Object', _constructor_object)

PyxelLoader.add_class_ref(['processor', 'class'])
PyxelLoader.add_class_ref(['processor', 'detector', 'class'])
PyxelLoader.add_class_ref(['processor', 'detector', None, 'class'])
PyxelLoader.add_class_ref(['processor', 'pipeline', 'class'])
PyxelLoader.add_class_ref(['processor', 'pipeline', 'class'])

PyxelLoader.add_class(ParametricConfig, ['parametric'])
PyxelLoader.add_class(StepValues, ['parametric', 'steps'], is_list=True)
PyxelLoader.add_class(ModelGroup, ['processor', 'pipeline', None])
PyxelLoader.add_class(ModelFunction, ['processor', 'pipeline', None, None])


# def _constructor_model_function(loader: PyxelLoader, node: yaml.MappingNode):
#     mapping = loader.construct_mapping(node)             # type: dict
#     model = ModelFunction(**mapping)
#     return model
#
#
# def _constructor_model_group(loader: PyxelLoader, node: yaml.SequenceNode):
#     if isinstance(node, yaml.ScalarNode):
#         return node.value
#
#     sequence = loader.construct_sequence(node)             # type: list
#     model = ModelGroup(sequence)
#     return model
#
#
# def _constructor_parametric(loader: PyxelLoader, node: yaml.MappingNode):
#     mapping = loader.construct_mapping(node, deep=True)     # type: dict
#     obj = ParametricConfig(**mapping)
#     return obj
#
#
# def _constructor_steps(loader: PyxelLoader, node: yaml.SequenceNode):
#     sequence = loader.construct_sequence(node, deep=True)     # type: list
#     obj = [StepValues(**kwargs) for kwargs in sequence]
#     return obj
#
# ClassConstructor(PyxelLoader, ParametricConfig, ['parametric'])
# ClassConstructor(PyxelLoader, StepValues, ['parametric', 'steps'], True)
# ClassConstructor(PyxelLoader, ModelGroup, ['processor', 'pipeline', None])
# ClassConstructor(PyxelLoader, ModelFunction, ['processor', 'pipeline', None, None])

# PyxelLoader.add_path_resolver('!ParametricConfig', ['parametric'])
# PyxelLoader.add_constructor('!ParametricConfig', _constructor_parametric)
#
# PyxelLoader.add_path_resolver('!StepValues', ['parametric', 'steps'])
# PyxelLoader.add_constructor('!StepValues', _constructor_steps)
#
# # NOTE 2: overrides the : 'processor', 'pipeline', 'class'
# PyxelLoader.add_path_resolver('!ModelGroup', ['processor', 'pipeline', None])
# PyxelLoader.add_constructor('!ModelGroup', _constructor_model_group)
#
# PyxelLoader.add_path_resolver('!Model', ['processor', 'pipeline', None, None])
# PyxelLoader.add_constructor('!Model', _constructor_model_function)
#
#
# def load(stream: t.Union[str, t.IO]):
#     """Parse a YAML document.
#
#     :param stream: document to process.
#     :return: a python object
#     """
#     return yaml.load(stream, Loader=PyxelLoader)
#
#
# def load_config(yaml_filename: Path):
#     """Load the YAML configuration file.
#
#     :param yaml_filename:
#     :return:
#     """
#     with yaml_filename.open('r') as file_obj:
#         cfg = load(file_obj)
#
#     return cfg

def load(yaml_file: t.Union[str, Path]):
    """TBW.

    :param yaml_file:
    :return:
    """
    if isinstance(yaml_file, str):
        with Path(yaml_file).open('r') as file_obj:
            return load_yaml(file_obj)
    else:
        with yaml_file.open('r') as file_obj:
            return load_yaml(file_obj)


def load_yaml(stream: t.Union[str, t.IO]):
    """Load a YAML document.

    :param stream: document to process.
    :return: a python object
    """
    result = yaml.load(stream, Loader=PyxelLoader)
    return result


def dump(cfg_obj) -> str:
    """Serialize a Python object into a YAML stream.

    :param cfg_obj: Object to serialize to YAML.
    :return: the YAML output as a `str`.
    """
    cfg_map = om.get_state_dict(cfg_obj)
    class_name = ''
    keys = []
    for paths in PyxelLoader.class_paths:
        key = ''
        class_name = paths[-1]
        key = '.'.join([str(path) for path in paths[:-1]])
        if '.None' in key:
            none_key = key.split('.None')[0]
            cfg_map_val = om.get_value(cfg_map, none_key)
            att_keys = cfg_map_val.keys()
            for att_key in att_keys:
                keys.append(key.replace('None', att_key))
        else:
            keys.append(key)

    # keys = [
    #     'processor',
    #     'processor.detector',
    #     'processor.detector.geometry',
    #     'processor.detector.environment',
    #     'processor.detector.characteristics',
    #     'processor.pipeline',
    # ]
    for key in keys:
        cfg_map_val = om.get_value(cfg_map, key)
        cfg_obj_val = om.get_value(cfg_obj, key)
        cfg_map_val[class_name] = cfg_obj_val.__module__ + '.' + cfg_obj_val.__class__.__name__

    result = yaml.dump(cfg_map, default_flow_style=False)
    return result

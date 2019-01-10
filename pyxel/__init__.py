"""Pyxel detector simulation framework."""
import os
import typing as t
import esapy_config as om

__all__ = ['models', 'detectors', 'pipelines',
           'attribute', 'detector_class',
           'check_type', 'check_path', 'check_range', 'check_choices',
           'validate_choices', 'validate_range']

__appname__ = 'Pyxel'
__author__ = 'David Lucsanyi'
__author_email__ = 'david.lucsanyi@esa.int'
__pkgname__ = 'pyxel'

__version__ = '0.3'


def detector_class(cls):
    """TBW."""
    return om.attr_class(cls)


def attribute(doc: t.Optional[str] = None,
              is_property: t.Optional[bool] = None,
              on_set: t.Optional[t.Callable] = None,
              on_get: t.Optional[t.Callable] = None,
              on_change: t.Optional[t.Callable] = None,
              use_dispatcher: t.Optional[bool] = None,
              on_get_update: t.Optional[bool] = None,
              **kwargs):
    """TBW."""
    return om.attr_def(doc, is_property, on_set, on_get, on_change,
                       use_dispatcher, on_get_update, **kwargs)


# from functools import wraps
def validate(func: t.Callable):
    """TBW."""
    # @wraps(func)
    # def new_func(*args, **kwargs):
    #     prev_func = om.validate(func)  # type: t.Callable
    #     return prev_func(*args, **kwargs)
    # return new_func
    new_func = om.validate(func)            # type: t.Callable
    new_func.__doc__ = func.__doc__                     # used by sphinx
    new_func.__annotations__ = func.__annotations__     # used by sphinx
    new_func.__module__ = func.__module__               # used by sphinx

    # new_func.__name__ = func.__name__               # not used by sphinx unless we missing something
    # new_func.__defaults__ = func.__defaults__       # not used by sphinx unless we missing something !!!!
    return new_func


def argument(name: str, **kwargs):
    """TBW."""
    return om.argument(name, **kwargs)


def check_type(att_type, is_optional: bool = False) -> t.Callable[..., bool]:
    """TBW."""
    return om.check_type_function(att_type, is_optional)


def check_path(path):
    """TBW."""
    return os.path.exists(path)


def check_range(min_val: t.Union[float, int], max_val: t.Union[float, int]):
                # step: t.Union[float, int] = None, enforce_step: bool = False):
    """TBW."""
    return om.check_range(min_val, max_val, step=None, enforce_step=False)
    # todo: rounding BUG in om.check_range() when value is a float!


def check_choices(choices: list):
    """TBW."""
    return om.check_choices(choices)


def validate_choices(choices, is_optional=False):
    """TBW."""
    return om.validate_choices(choices, is_optional)


def validate_range(min_val: t.Union[float, int], max_val: t.Union[float, int],
                   # step: t.Union[float, int] = None, enforce_step: bool = True,
                   is_optional: bool = False):
    """TBW."""
    return om.validate_range(min_val, max_val, is_optional)
    # todo: rounding BUG in om.check_range() when value is a float!


def validate_type(att_type, is_optional: bool = False):
    """TBW."""
    return om.validate_type(att_type, is_optional)


def pyxel_yaml_loader():
    """TBW."""
    from pyxel.pipelines.parametric import Configuration
    from pyxel.pipelines.parametric import ParametricAnalysis
    from pyxel.pipelines.parametric import StepValues
    from pyxel.calibration.calibration import Calibration
    from pyxel.calibration.calibration import Algorithm
    from pyxel.pipelines.model_function import ModelFunction
    from pyxel.pipelines.model_group import ModelGroup

    om.ObjectModelLoader.add_class_ref(['detector', 'class'])
    om.ObjectModelLoader.add_class_ref(['detector', None, 'class'])

    om.ObjectModelLoader.add_class_ref(['pipeline', 'class'])
    om.ObjectModelLoader.add_class(ModelGroup, ['pipeline', None])
    om.ObjectModelLoader.add_class(ModelFunction, ['pipeline', None, None])

    om.ObjectModelLoader.add_class(Configuration, ['simulation'])

    om.ObjectModelLoader.add_class(ParametricAnalysis, ['simulation', 'parametric_analysis'])
    om.ObjectModelLoader.add_class(StepValues, ['simulation', 'parametric_analysis', 'steps'], is_list=True)

    om.ObjectModelLoader.add_class(Calibration, ['simulation', 'calibration'])
    om.ObjectModelLoader.add_class(Algorithm, ['simulation', 'calibration', 'algorithm'])
    om.ObjectModelLoader.add_class(ModelFunction, ['simulation', 'calibration', 'fitness_function'])


pyxel_yaml_loader()

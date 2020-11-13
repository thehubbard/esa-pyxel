#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#

import yaml
from yaml import Loader
import typing as t
from pathlib import Path
from pyxel.evaluator import evaluate_reference
from functools import partial
from pyxel.pipelines import DetectionPipeline, ModelFunction, ModelGroup
import attr

from pyxel.detectors import (
    CCD,
    CMOS,
    Environment,
    CCDGeometry,
    CCDCharacteristics,
    CMOSGeometry,
    CMOSCharacteristics,
    Material,
)
from pyxel.single import Single
from pyxel.parametric import Parametric
from pyxel.calibration import Calibration
from pyxel.dynamic import Dynamic
from pyxel.util import Outputs


def build_callable(func: str, arguments: t.Optional[dict] = None) -> t.Callable:
    """Create a callable.

    Parameters
    ----------
    func
    arguments

    Returns
    -------
    callable
        TBW.
    """
    assert isinstance(func, str)
    assert arguments is None or isinstance(arguments, dict)

    if arguments is None:
        arguments = {}

    func_callable = evaluate_reference(func)  # type: t.Callable

    return partial(func_callable, **arguments)


def to_outputs(dct) -> Outputs:
    return Outputs(dct)


def to_single(dct) -> Single:
    return Single(outputs=to_outputs(dct["outputs"]))


def to_ccd_geometry(dct: dict) -> CCDGeometry:
    return CCDGeometry(**dct)


def to_cmos_geometry(dct: dict) -> CMOSGeometry:
    return CMOSGeometry(**dct)


def to_material(dct: dict) -> Material:
    return Material(**dct)


def to_environment(dct: dict) -> Environment:
    return Environment(**dct)


def to_ccd_characteristics(dct: dict) -> CCDCharacteristics:
    return CCDCharacteristics(**dct)


def to_cmos_characteristics(dct: dict) -> CMOSCharacteristics:
    return CMOSCharacteristics(**dct)


def to_ccd(dct: dict) -> CCD:
    return CCD(
        geometry=to_ccd_geometry(dct["geometry"]),
        material=to_material(dct["material"]),
        environment=to_environment(dct["environment"]),
        characteristics=to_ccd_characteristics(dct["characteristics"]),
    )


def to_cmos(dct: dict) -> CMOS:
    return CMOS(
        geometry=to_cmos_geometry(dct["geometry"]),
        material=to_material(dct["material"]),
        environment=to_environment(dct["environment"]),
        characteristics=to_cmos_characteristics(dct["characteristics"]),
    )


def to_model_function(dct: dict) -> ModelFunction:
    func = dct["func"]
    name = dct["name"]
    try:
        arguments = dct["arguments"]
    except:
        arguments = {}
    enabled = dct["enabled"]

    assert isinstance(func, str)
    assert isinstance(name, str)
    assert isinstance(arguments, dict)
    assert isinstance(enabled, bool)

    func_callable = evaluate_reference(func)  # type: t.Callable

    return ModelFunction(
        func=func_callable, name=name, arguments=arguments, enabled=enabled
    )


def to_model_group(models_list: t.Sequence[dict]) -> ModelGroup:
    models = []
    for model_dict in models_list:
        models.append(to_model_function(model_dict))
    return ModelGroup(models=models)


def to_pipeline(dct: dict) -> DetectionPipeline:
    pipeline_dict = {}
    for model_group_name in dct.keys():
        pipeline_dict.update({model_group_name: to_model_group(dct[model_group_name])})
    return DetectionPipeline(**pipeline_dict)


@attr.s
class Configuration:
    single: Single = attr.ib(init=False)
    parametric: Parametric = attr.ib(init=False)
    calibration: Calibration = attr.ib(init=False)
    dynamic: Dynamic = attr.ib(init=False)
    ccd_detector: CCD = attr.ib(init=False)
    cmos_detector: CMOS = attr.ib(init=False)
    pipeline: DetectionPipeline = attr.ib(init=False)


def build_configuration(dct: dict) -> Configuration:
    configuration = Configuration()

    if "single" in dct:
        configuration.single = to_single(dct["single"])
    elif "parametric" in dct:
        configuration.parametric = to_parametric(dct["parametric"])
    elif "calibration" in dct:
        configuration.calibration = to_calibration(dct["calibration"])
    elif "dynamic" in dct:
        configuration.dynamic = to_dynamic(dct["dynamic"])
    else:
        raise (ValueError)

    if "ccd_detector" in dct:
        configuration.ccd_detector = to_ccd(dct["ccd_detector"])
    elif "cmos_detector" in dct:
        configuration.cmos_detector = to_cmos(dct["cmos_detector"])
    else:
        raise (ValueError)

    configuration.pipeline = to_pipeline(dct["pipeline"])

    return configuration


def load(yaml_file: t.Union[str, Path]) -> t.Any:
    """Load YAML file.
    :param yaml_file:
    :return:
    """
    filename = Path(yaml_file).resolve()
    if not filename.exists():
        raise FileNotFoundError(f"Cannot find configuration file '{filename}'.")
    with filename.open("r") as file_obj:
        return load_yaml(file_obj)


def load_yaml(stream: t.Union[str, t.IO]) -> t.Any:
    """Load a YAML document.

    :param stream: document to process.
    :return: a python object
    """
    result = yaml.load(stream, Loader=Loader)
    return result


if __name__ == "__main__":
    configuration = Configuration()
    cfg = load("../../examples/single.yaml")
    print(cfg)
    detector = to_ccd(cfg["ccd_detector"])
    pipeline = to_pipeline(cfg["pipeline"])
    single = to_single(cfg["single"])
    print(3)
    pass

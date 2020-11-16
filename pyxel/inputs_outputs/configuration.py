#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#

import typing as t
from functools import partial
from pathlib import Path

import attr
import yaml

from ..calibration import Algorithm, Calibration
from ..detectors import (
    CCD,
    CMOS,
    CCDCharacteristics,
    CCDGeometry,
    CMOSCharacteristics,
    CMOSGeometry,
    Environment,
    Material,
)
from ..dynamic import Dynamic
from ..evaluator import evaluate_reference
from .calibration_outputs import (
    CalibrationOutputs,
    CalibrationPlot,
    ChampionsPlot,
    FittingPlot,
    PopulationPlot,
)
from .dynamic_outputs import DynamicOutputs
from .outputs import PlotArguments
from .parametric_outputs import ParametricOutputs, ParametricPlot
from .single_outputs import SingleOutputs, SinglePlot
from ..parametric import ParameterValues, Parametric
from ..pipelines import DetectionPipeline, ModelFunction, ModelGroup
from ..single import Single


@attr.s
class Configuration:
    pipeline: DetectionPipeline = attr.ib(init=False)
    single: t.Optional[Single] = attr.ib(default=None)
    parametric: t.Optional[Parametric] = attr.ib(default=None)
    calibration: t.Optional[Calibration] = attr.ib(default=None)
    dynamic: t.Optional[Dynamic] = attr.ib(default=None)
    ccd_detector: t.Optional[CCD] = attr.ib(default=None)
    cmos_detector: t.Optional[CMOS] = attr.ib(default=None)


def to_plot_arguments(dct: dict) -> t.Optional[PlotArguments]:
    if dct is None:
        return None
    return PlotArguments(**dct)


def to_single_plot(dct: dict) -> t.Optional[SinglePlot]:
    if dct is None:
        return None
    dct.update({"plot_args": to_plot_arguments(dct["plot_args"])})
    return SinglePlot(**dct)


def to_single_outputs(dct: dict) -> SingleOutputs:
    dct.update({"single_plot": to_single_plot(dct["single_plot"])})
    return SingleOutputs(**dct)


def to_single(dct: dict) -> Single:
    return Single(outputs=to_single_outputs(dct["outputs"]))


# TODO: Dynamic uses single plot for now
def to_dynamic_outputs(dct: dict) -> DynamicOutputs:
    dct.update({"single_plot": to_single_plot(dct["single_plot"])})
    return DynamicOutputs(**dct)


def to_dynamic(dct: dict) -> Dynamic:
    dct.update({"outputs": to_dynamic_outputs(dct["outputs"])})
    return Dynamic(**dct)


def to_parametric_plot(dct: dict) -> t.Optional[ParametricPlot]:
    if dct is None:
        return None
    dct.update({"plot_args": to_plot_arguments(dct["plot_args"])})
    return ParametricPlot(**dct)


def to_parametric_outputs(dct: dict) -> ParametricOutputs:
    dct.update({"parametric_plot": to_parametric_plot(dct["parametric_plot"])})
    return ParametricOutputs(**dct)


def to_parameters(dct: dict) -> ParameterValues:
    return ParameterValues(**dct)


def to_parametric(dct: dict) -> Parametric:
    dct.update(
        {"parameters": [to_parameters(param_dict) for param_dict in dct["parameters"]]}
    )
    dct.update({"outputs": to_parametric_outputs(dct["outputs"])})
    return Parametric(**dct)


def to_champions_plot(dct: dict) -> t.Optional[ChampionsPlot]:
    if dct is None:
        return None
    dct.update({"plot_args": to_plot_arguments(dct["plot_args"])})
    return ChampionsPlot(**dct)


def to_population_plot(dct: dict) -> t.Optional[PopulationPlot]:
    if dct is None:
        return None
    dct.update({"plot_args": to_plot_arguments(dct["plot_args"])})
    return PopulationPlot(**dct)


def to_fitting_plot(dct: dict) -> t.Optional[FittingPlot]:
    if dct is None:
        return None
    dct.update({"plot_args": to_plot_arguments(dct["plot_args"])})
    return FittingPlot(**dct)


def to_calibration_plot(dct: dict) -> t.Optional[CalibrationPlot]:
    if "champions_plot" in dct:
        dct.update({"champions_plot": to_champions_plot(dct["champions_plot"])})
    if "population_plot" in dct:
        dct.update({"population_plot": to_population_plot(dct["population_plot"])})
    if "fitting_plot" in dct:
        dct.update({"fitting_plot": to_fitting_plot(dct["fitting_plot"])})
    if dct is None:
        return None
    return CalibrationPlot(**dct)


def to_calibration_outputs(dct: dict):
    dct.update({"calibration_plot": to_calibration_plot(dct["calibration_plot"])})
    return CalibrationOutputs(**dct)


def to_algorithm(dct: dict) -> t.Optional[Algorithm]:
    if dct is None:
        return None
    return Algorithm(**dct)


def to_callable(dct: dict) -> t.Callable:
    func = evaluate_reference(dct["func"])
    arguments = dct["arguments"]
    if arguments is None:
        arguments = {}
    return partial(func, **arguments)


def to_calibration(dct: dict) -> Calibration:
    dct.update({"outputs": to_calibration_outputs(dct["outputs"])})
    dct.update({"fitness_function": to_callable(dct["fitness_function"])})
    dct.update({"algorithm": to_algorithm(dct["algorithm"])})
    dct.update(
        {"parameters": [to_parameters(param_dict) for param_dict in dct["parameters"]]}
    )
    return Calibration(**dct)


def to_ccd_geometry(dct: dict) -> t.Optional[CCDGeometry]:
    if dct is None:
        return None
    return CCDGeometry(**dct)


def to_cmos_geometry(dct: dict) -> t.Optional[CMOSGeometry]:
    if dct is None:
        return None
    return CMOSGeometry(**dct)


def to_material(dct: dict) -> t.Optional[Material]:
    if dct is None:
        return None
    return Material(**dct)


def to_environment(dct: dict) -> t.Optional[Environment]:
    if dct is None:
        return None
    return Environment(**dct)


def to_ccd_characteristics(dct: dict) -> t.Optional[CCDCharacteristics]:
    if dct is None:
        return None
    return CCDCharacteristics(**dct)


def to_cmos_characteristics(dct: dict) -> t.Optional[CMOSCharacteristics]:
    if dct is None:
        return None
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
    dct.update({"func": evaluate_reference(dct["func"])})
    return ModelFunction(**dct)


def to_model_group(models_list: t.Sequence[dict]) -> t.Optional[ModelGroup]:
    if models_list is None:
        return None
    models = [to_model_function(model_dict) for model_dict in models_list]
    return ModelGroup(models=models)


def to_pipeline(dct: dict) -> DetectionPipeline:
    for model_group_name in dct.keys():
        dct.update({model_group_name: to_model_group(dct[model_group_name])})
    return DetectionPipeline(**dct)


def build_configuration(dct: dict) -> Configuration:

    configuration = Configuration()  # type: Configuration

    configuration.pipeline = to_pipeline(dct["pipeline"])

    if "single" in dct:
        configuration.single = to_single(dct["single"])
    elif "parametric" in dct:
        configuration.parametric = to_parametric(dct["parametric"])
    elif "calibration" in dct:
        configuration.calibration = to_calibration(dct["calibration"])
    elif "dynamic" in dct:
        configuration.dynamic = to_dynamic(dct["dynamic"])
    else:
        raise (ValueError("No mode configuration provided."))

    if "ccd_detector" in dct:
        configuration.ccd_detector = to_ccd(dct["ccd_detector"])
    elif "cmos_detector" in dct:
        configuration.cmos_detector = to_cmos(dct["cmos_detector"])
    else:
        raise (ValueError("No detector configuration provided."))

    return configuration


def load(yaml_file: t.Union[str, Path]) -> Configuration:
    """Load YAML file.
    :param yaml_file:
    :return:
    """
    filename = Path(yaml_file).resolve()
    if not filename.exists():
        raise FileNotFoundError(f"Cannot find configuration file '{filename}'.")
    with filename.open("r") as file_obj:
        return build_configuration(load_yaml(file_obj))


def load_yaml(stream: t.Union[str, t.IO]) -> t.Any:
    """Load a YAML document.

    :param stream: document to process.
    :return: a python object
    """
    result = yaml.load(stream, Loader=yaml.SafeLoader)
    return result


#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""
from .object_model import ObjectModelLoader
from .object_model import load  # noqa: F401
import typing as t
from functools import partial
from pyxel.evaluator import evaluate_reference


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


# TODO: Re-develop the YAML loader and representer. See Issue #59.
def pyxel_yaml_loader():
    """TBW."""
    from pyxel.parametric.parametric import Configuration
    from pyxel.parametric.parametric import ParametricAnalysis
    from pyxel.parametric.parameter_values import ParameterValues
    from pyxel.pipelines import ModelFunction
    from pyxel.pipelines import ModelGroup
    from pyxel.util import Outputs

    try:
        from pyxel.calibration.calibration import Calibration
        from pyxel.calibration.calibration import Algorithm

        ObjectModelLoader.add_class(Calibration, ["simulation", "calibration"])
        ObjectModelLoader.add_class(
            Algorithm, ["simulation", "calibration", "algorithm"]
        )
        ObjectModelLoader.add_class(
            build_callable, ["simulation", "calibration", "fitness_function"]
        )
        ObjectModelLoader.add_class(
            ParameterValues, ["simulation", "calibration", "parameters"], is_list=True
        )
        ObjectModelLoader.add_class(
            ParameterValues,
            ["simulation", "calibration", "result_input_arguments"],
            is_list=True,
        )
        # WITH_CALIBRATION = True  # noqa: N806
    except ImportError:
        # WITH_CALIBRATION = False  # noqa: N806
        pass

    from pyxel.detectors import CCD, CMOS, Detector
    from pyxel.detectors import Geometry, Characteristics, Material, Environment
    from pyxel.detectors.ccd import CCDGeometry, CCDCharacteristics
    from pyxel.detectors.cmos import CMOSGeometry, CMOSCharacteristics

    from pyxel.pipelines import DetectionPipeline

    ObjectModelLoader.add_class(CCD, ["ccd_detector"])  # pyxel.detectors.ccd.CCD
    ObjectModelLoader.add_class(CMOS, ["cmos_detector"])
    ObjectModelLoader.add_class(Detector, ["detector"])

    ObjectModelLoader.add_class(CCDGeometry, ["ccd_detector", "geometry"])
    ObjectModelLoader.add_class(CMOSGeometry, ["cmos_detector", "geometry"])
    ObjectModelLoader.add_class(Geometry, ["detector", "geometry"])

    ObjectModelLoader.add_class(CCDCharacteristics, ["ccd_detector", "characteristics"])
    ObjectModelLoader.add_class(
        CMOSCharacteristics, ["cmos_detector", "characteristics"]
    )
    ObjectModelLoader.add_class(Characteristics, ["detector", "characteristics"])

    ObjectModelLoader.add_class(Material, [None, "material"])
    ObjectModelLoader.add_class(Environment, [None, "environment"])

    ObjectModelLoader.add_class(DetectionPipeline, ["pipeline"])

    ObjectModelLoader.add_class(ModelGroup, ["pipeline", None])
    ObjectModelLoader.add_class(ModelFunction, ["pipeline", None, None])

    ObjectModelLoader.add_class(Configuration, ["simulation"])

    ObjectModelLoader.add_class(Outputs, ["simulation", "outputs"])

    ObjectModelLoader.add_class(ParametricAnalysis, ["simulation", "parametric"])
    ObjectModelLoader.add_class(
        ParameterValues, ["simulation", "parametric", "parameters"], is_list=True
    )


pyxel_yaml_loader()  # TODO: avoid auto-calling functions on import.

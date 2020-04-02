#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""
import itertools
import logging
import typing as t
from copy import deepcopy
from enum import Enum

import numpy as np

from pyxel.parametric.parameter_values import ParameterValues
from pyxel.state import get_obj_att, get_value

if t.TYPE_CHECKING:
    from ..pipelines import Processor
    from ..calibration.calibration import Calibration
    from ..util import Outputs


class ParametricMode(Enum):
    """TBW."""

    Embedded = "embedded"
    Sequential = "sequential"
    Parallel = "parallel"


# TODO: Use `Enum` for `parametric_mode` ?
class ParametricAnalysis:
    """TBW."""

    def __init__(
        self,
        parametric_mode: str,
        parameters: t.List[ParameterValues],
        from_file: t.Optional[str] = None,
        column_range: t.Optional[t.Tuple[int, int]] = None,
    ):
        """TBW."""
        self.parametric_mode = ParametricMode(parametric_mode)  # type: ParametricMode
        self._parameters = parameters
        self.file = from_file
        self.data = None  # type: t.Optional[np.ndarray]
        if column_range:
            self.columns = slice(*column_range)

    def __repr__(self):
        """TBW."""
        cls_name = self.__class__.__name__  # type: str
        return f"{cls_name}<mode={self.parametric_mode!s}>"

    @property
    def enabled_steps(self) -> t.List[ParameterValues]:
        """TBW."""
        return [step for step in self._parameters if step.enabled]

    def _parallel(self, processor: "Processor") -> "t.Iterator[Processor]":
        """TBW.

        :param processor:
        :return:
        """
        self.data = np.loadtxt(self.file)[:, self.columns]
        for data_array in self.data:
            i = 0
            new_proc = deepcopy(processor)  # type: Processor
            for step in self.enabled_steps:
                key = step.key

                # TODO: this is confusing code. Fix this.
                #       Furthermore 'step.values' should be a `t.List[int, float]` and not a `str`
                if step.values == "_":
                    value = data_array[i]
                    i += 1

                elif isinstance(step.values, list) and all(
                    x == "_" for x in step.values[:]
                ):
                    value = data_array[i : i + len(step.values)]  # noqa: E203
                    i += len(value)

                else:
                    raise ValueError(
                        'Only "_" characters (or a list of them) should be used to '
                        "indicate parameters updated from file in parallel"
                    )

                new_proc.set(key, value)
            yield new_proc

    def _sequential(self, processor: "Processor") -> "t.Iterator[Processor]":
        """TBW.

        :param processor:
        :return:
        """
        for step in self.enabled_steps:  # type: ParameterValues
            key = step.key  # type : str
            for value in step:
                # step.current = value
                new_proc = deepcopy(processor)  # type: Processor
                new_proc.set(key=key, value=value)
                yield new_proc

    def _embedded(self, processor: "Processor") -> "t.Iterator[Processor]":
        """TBW.

        :param processor:
        :return:
        """
        all_steps = self.enabled_steps
        keys = [step.key for step in self.enabled_steps]
        for params in itertools.product(*all_steps):
            new_proc = deepcopy(processor)  # type: Processor
            for key, value in zip(keys, params):
                # for step in all_steps:
                #     if step.key == key:
                #         step.current = value
                new_proc.set(key=key, value=value)
            yield new_proc

    def collect(self, processor: "Processor") -> "t.Iterator[Processor]":
        """TBW."""
        for step in self.enabled_steps:  # type: ParameterValues

            # TODO: the string literal expressions are difficult to maintain.
            #     Example: 'pipeline.', '.arguments', '.enabled'
            #     We may want to consider an API for this.
            # Proposed API:
            # value = operator.attrgetter(step.key)(processor)
            if "pipeline." in step.key:
                model_name = step.key[: step.key.find(".arguments")]  # type: str
                model_enabled = model_name + ".enabled"  # type: str
                if not processor.get(model_enabled):
                    raise ValueError(
                        f"The '{model_name}' model referenced in parametric configuration "
                        f"has not been enabled in yaml config!"
                    )

            if (
                any(x == "_" for x in step.values[:])
                and self.parametric_mode != ParametricMode.Parallel
            ):
                raise ValueError(
                    "Either define 'parallel' as parametric mode or "
                    "do not use '_' character in 'values' field"
                )

        if self.parametric_mode == ParametricMode.Embedded:
            configs_it = self._embedded(processor)  # type: t.Iterator[Processor]

        elif self.parametric_mode == ParametricMode.Sequential:
            configs_it = self._sequential(processor)

        elif self.parametric_mode == ParametricMode.Parallel:
            configs_it = self._parallel(processor)

        else:
            # configs_it = iter([])
            raise NotImplementedError()

        return configs_it

    def debug(self, processor: "Processor") -> list:
        """TBW."""
        result = []
        configs = self.collect(processor)
        for i, config in enumerate(configs):
            values = []
            for step in self.enabled_steps:
                _, att = get_obj_att(config, step.key)
                value = get_value(config, step.key)
                values.append((att, value))
            logging.debug("%d: %r" % (i, values))
            result.append((i, values))
        return result


# TODO: Use a `Enum` for 'mode' ?
# TODO: Create several classes `ConfigurationSingle`, `ConfigurationParametric`,
#       `ConfigurationCalibration` and `ConfigurationDynamic`
# TODO: Move this class into its own file 'configuration.py'
class Configuration:
    """TBW."""

    def __init__(
        self,
        mode: str,
        outputs: "Outputs",
        parametric: t.Optional[ParametricAnalysis] = None,
        calibration: "t.Optional[Calibration]" = None,
        dynamic: t.Optional[t.Dict[str, t.Any]] = None,
    ):
        """TBW.

        Parameters
        ----------
        mode
        outputs
        parametric
        calibration
        dynamic
        """
        if mode not in ["single", "parametric", "calibration", "dynamic"]:
            raise ValueError(
                "Non-existing running mode defined for Pyxel in yaml config file."
            )

        self.mode = mode  # type: str

        self.outputs = outputs  # type: Outputs

        self.parametric = parametric  # type: t.Optional[ParametricAnalysis]
        self.calibration = calibration  # type: t.Optional[Calibration]
        self.dynamic = dynamic  # type: t.Optional[t.Dict[str, t.Any]]

        if mode == "parametric":
            assert self.parametric
            self.outputs.params_func(self.parametric)

    def __repr__(self) -> str:
        """TBW."""
        cls_name = self.__class__.__name__  # type: str

        return f"{cls_name}<mode={self.mode!r}, outputs={self.outputs!r}>"

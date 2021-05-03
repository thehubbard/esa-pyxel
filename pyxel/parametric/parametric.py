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
from tqdm.auto import tqdm
from dask import delayed
import dask

import numpy as np

from pyxel.parametric.parameter_values import ParameterValues
from pyxel.state import get_obj_att, get_value

if t.TYPE_CHECKING:
    from ..inputs_outputs import ParametricOutputs
    from ..pipelines import Processor


class ParametricMode(Enum):
    """TBW."""

    Embedded = "embedded"
    Sequential = "sequential"
    Parallel = "parallel"


# TODO: Use `Enum` for `parametric_mode` ?
class Parametric:
    """TBW."""

    def __init__(
        self,
        outputs: "ParametricOutputs",
        mode: str,
        parameters: t.Sequence[ParameterValues],
        from_file: t.Optional[str] = None,
        column_range: t.Optional[t.Tuple[int, int]] = None,
        with_dask: bool = False,
    ):
        self.outputs = outputs
        self.parametric_mode = ParametricMode(mode)  # type: ParametricMode
        self._parameters = parameters
        self.file = from_file
        self.data = None  # type: t.Optional[np.ndarray]
        if column_range:
            self.columns = slice(*column_range)
        self.with_dask = with_dask

    def __repr__(self):
        cls_name = self.__class__.__name__  # type: str
        return f"{cls_name}<mode={self.parametric_mode!s}>"

    @property
    def enabled_steps(self) -> t.Sequence[ParameterValues]:
        """TBW."""
        return [step for step in self._parameters if step.enabled]

    def _parallel(self, processor: "Processor") -> "t.Iterator[Processor]":
        """TBW.

        :param processor:
        :return:
        """
        result = np.loadtxt(self.file)[:, self.columns]  # type: np.ndarray
        self.data = result
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

    def run_parametric(self, processor: Processor) -> None:
        """TBW."""

        # Check if all keys from 'parametric' are valid keys for object 'pipeline'
        for param_value in self.enabled_steps:
            key = param_value.key  # type: str
            assert processor.has(key)

        processors_it = self.collect(processor)  # type: t.Iterator[Processor]

        result_list = []  # type: t.List[Result]
        output_filenames = []  # type: t.List[t.Sequence[Path]]

        # Run all pipelines
        for proc in tqdm(processors_it):  # type: Processor

            if not self.with_dask:
                result_proc = proc.run_pipeline()  # type: Processor
                result_val = self.outputs.extract_func(
                    processor=result_proc
                )  # type: Result

                # filenames = parametric_outputs.save_to_file(
                #    processor=result_proc
                # )  # type: t.Sequence[Path]

            else:
                result_proc = delayed(proc.run_pipeline)()
                result_val = delayed(self.outputs.extract_func)(processor=result_proc)

                # filenames = delayed(parametric_outputs.save_to_file)(processor=result_proc)

            result_list.append(result_val)
            # output_filenames.append(filenames)  # TODO: This is not used

        if not self.with_dask:
            plot_array = self.outputs.merge_func(result_list)  # type: np.ndarray
        else:
            array = delayed(self.outputs.merge_func)(result_list)
            plot_array = dask.compute(array)

        # TODO: Plot with dask ?
        # if parametric_outputs.parametric_plot is not None:
        #    parametric_outputs.plotting_func(plot_array)

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

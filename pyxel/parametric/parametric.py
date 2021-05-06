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
from pyxel.inputs_outputs.loader import load_table
from typing_extensions import Literal
import xarray as xr
import operator

import numpy as np

from pyxel.parametric.parameter_values import ParameterValues
from pyxel.state import get_obj_att, get_value

if t.TYPE_CHECKING:
    from ..inputs_outputs import ParametricOutputs
    from ..pipelines import Processor


def create_new_processor(processor: "Processor", parameter_dict: dict) -> "Processor":

    new_processor = deepcopy(processor)

    for key in parameter_dict.keys():
        new_processor.set(key=key, value=parameter_dict[key])

    return new_processor


class ParametricMode(Enum):
    """TBW."""

    Embedded = "embedded"
    Sequential = "sequential"
    Parallel = "parallel"

class ResultType(Enum):
    """TBW."""

    Image = "image"
    Signal = "signal"
    Pixel = "pixel"


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
        result_type: Literal["image", "signal", "pixel"] = "image",
    ):
        self.outputs = outputs
        self.parametric_mode = ParametricMode(mode)  # type: ParametricMode
        self._parameters = parameters
        self.file = from_file
        self.data = None  # type: t.Optional[np.ndarray]
        if column_range:
            self.columns = slice(*column_range)
        self.with_dask = with_dask
        self.parameter_types = {}

    def __repr__(self):
        cls_name = self.__class__.__name__  # type: str
        return f"{cls_name}<mode={self.parametric_mode!s}>"

    @property
    def enabled_steps(self) -> t.Sequence[ParameterValues]:
        """TBW."""
        return [step for step in self._parameters if step.enabled]

    def _parallel_parameters(self) -> "t.Iterator[Processor]":
        """TBW.

        :param processor:
        :return:
        """
        # TODO: is self.data really needed
        result = load_table(self.file)[:, self.columns]  # type: np.ndarray
        self.data = result
        for data_array in self.data:
            i = 0
            parameters = {}
            for step in self.enabled_steps:
                key = step.key

                # TODO: this is confusing code. Fix this.
                #       Furthermore 'step.values' should be a `t.List[int, float]` and not a `str`
                if step.values == "_":
                    value = data_array[i]
                    i += 1
                    parameters.update({key: value})

                elif isinstance(step.values, list):

                    values = np.asarray(deepcopy(step.values))  # type: np.ndarray
                    sh = values.shape  # type: tuple
                    values_flattened = values.flatten()

                    if all(x == "_" for x in values_flattened):
                        value = data_array[i : i + len(values_flattened)]  # noqa: E203
                        i += len(value)
                        value = value.reshape(sh).tolist()
                        parameters.update({key: value})
                    else:
                        raise ValueError(
                            'Only "_" characters (or a list of them) should be used to '
                            "indicate parameters updated from file in parallel"
                        )
            yield parameters

    def _sequential_parameters(self) -> "t.Iterator[dict]":
        """TBW.

        :param processor:
        :return:
        """
        for step in self.enabled_steps:  # type: ParameterValues
            key = step.key  # type : str
            for value in step:
                yield {key: value}

    def _embedded_parameters(self) -> "t.Iterator[dict]":
        """TBW.

        :param processor:
        :return:
        """
        all_steps = self.enabled_steps
        keys = [step.key for step in self.enabled_steps]
        for params in itertools.product(*all_steps):
            parameters = {}
            for key, value in zip(keys, params):
                parameters.update({key: value})
            yield parameters

    def _parameter_it(self) -> t.Callable:
        if self.parametric_mode == ParametricMode.Embedded:
            return self._embedded_parameters

        elif self.parametric_mode == ParametricMode.Sequential:
            return self._sequential_parameters

        elif self.parametric_mode == ParametricMode.Parallel:
            return self._parallel_parameters

    def _processors_it(self, processor: "Processor"):

        parameter_it = self._parameter_it()

        for parameter_dict in parameter_it():
            yield create_new_processor(processor=processor, parameter_dict=parameter_dict), parameter_dict

    def _delayed_processors(self, processor: "Processor"):

        processors = []
        delayed_processor = delayed(processor)
        parameter_it = self._parameter_it()

        for parameter_dict in parameter_it():
            delayed_parameter_dict = delayed(parameter_dict)
            processors.append(delayed(create_new_processor)(processor=delayed_processor, parameter_dict=delayed_parameter_dict))

        return processors

    def _get_parameter_types(self):
        for step in self.enabled_steps:
            self.parameter_types.update({step.key: step.type})
        return self.parameter_types

    def run_parametric(self, processor: "Processor") -> None:
        """TBW."""

        for step in self.enabled_steps:  # type: ParameterValues

            key = step.key  # type: str
            assert processor.has(key)

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
                        f"The '{model_name}' model referenced in parametric configuration"
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

        result_list = []
        datasets = []
        output_array_str = "detector.image.array"

        if self.with_dask:

            delayed_processor_list = self._delayed_processors(processor)

            for proc in delayed_processor_list:
                result_proc = delayed(proc.run_pipeline())
                result_list.append(result_proc)

            out = []

        else:
            i = 0
            for proc, parameter_dict in tqdm(self._processors_it(processor)):
                result_proc = proc.run_pipeline()
                self.outputs.save_to_file(result_proc)

                ds = xr.Dataset()

                #for output_array_str in self.output.save_dataset:

                output_array = operator.attrgetter(output_array_str)(result_proc)
                rows, columns = (result_proc.detector.geometry.row, result_proc.detector.geometry.row)

                da = xr.DataArray(name=output_array_str.split('.')[-2], data=output_array, dims=['x', 'y'], coords={'x': range(rows), 'y': range(columns), "i": [i]})

                datasets.append(da)

                #result_list.append(result_proc)

            #out = xr.concat(datasets, "i").assign_coords({'i': range(len(list(self._parameter_it()())))})
            out = xr.combine_by_coords(datasets)

        return out

        # processors_it = self.collect(processor)  # type: t.Iterator[Processor]
        #
        # result_list = []  # type: t.List[Result]
        # output_filenames = []  # type: t.List[t.Sequence[Path]]
        #
        # # Run all pipelines
        # for proc in tqdm(processors_it):  # type: Processor
        #
        #     if not self.with_dask:
        #         result_proc = proc.run_pipeline()  # type: Processor
        #         result_val = self.outputs.extract_func(
        #             processor=result_proc
        #         )  # type: Result
        #
        #         # filenames = parametric_outputs.save_to_file(
        #         #    processor=result_proc
        #         # )  # type: t.Sequence[Path]
        #
        #     else:
        #         result_proc = delayed(proc.run_pipeline)()
        #         result_val = delayed(self.outputs.extract_func)(processor=result_proc)
        #
        #         # filenames = delayed(parametric_outputs.save_to_file)(processor=result_proc)
        #
        #     result_list.append(result_val)
        #     # output_filenames.append(filenames)  # TODO: This is not used
        #
        # if not self.with_dask:
        #     plot_array = self.outputs.merge_func(result_list)  # type: np.ndarray
        # else:
        #     array = delayed(self.outputs.merge_func)(result_list)
        #     plot_array = dask.compute(array)
        #
        # # TODO: Plot with dask ?
        # # if parametric_outputs.parametric_plot is not None:
        # #    parametric_outputs.plotting_func(plot_array)

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

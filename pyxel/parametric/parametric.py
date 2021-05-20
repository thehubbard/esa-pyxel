#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""
import itertools
import operator

import logging
import typing as t
from copy import deepcopy
from enum import Enum

import numpy as np
import xarray as xr
from dask import delayed
from tqdm.auto import tqdm
from typing_extensions import Literal

# import dask
from pyxel.inputs_outputs.loader import load_table
from pyxel.parametric.parameter_values import ParameterType, ParameterValues
from pyxel.state import get_obj_att, get_value

if t.TYPE_CHECKING:
    from ..inputs_outputs import ParametricOutputs
    from ..pipelines import Processor


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


class Result(t.NamedTuple):
    dataset: t.Union[xr.Dataset, t.Dict[str, xr.Dataset]]
    parameters: xr.Dataset
    logs: xr.Dataset


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
        self.parameter_types = {}  # type: dict

    def __repr__(self):
        cls_name = self.__class__.__name__  # type: str
        return f"{cls_name}<mode={self.parametric_mode!s}>"

    @property
    def enabled_steps(self) -> t.Sequence[ParameterValues]:
        """TBW."""
        return [step for step in self._parameters if step.enabled]

    def _parallel_parameters(self) -> t.Generator[t.Tuple[int, dict], None, None]:
        """TBW.

        :param processor:
        :return:
        """
        # TODO: is self.data really needed
        if self.file is not None:
            result = load_table(self.file).to_numpy()[
                :, self.columns
            ]  # type: np.ndarray
        else:
            raise ValueError("File for parallel parametric mode not specified!")
        self.data = result
        for index, data_array in enumerate(self.data):
            i = 0
            parameter_dict = {}
            for step in self.enabled_steps:
                key = step.key

                # TODO: this is confusing code. Fix this.
                #       Furthermore 'step.values' should be a `t.List[int, float]` and not a `str`
                if step.values == "_":
                    value = data_array[i]
                    i += 1
                    parameter_dict.update({key: value})

                elif isinstance(step.values, list):

                    values = np.asarray(deepcopy(step.values))  # type: np.ndarray
                    sh = values.shape  # type: tuple
                    values_flattened = values.flatten()

                    if all(x == "_" for x in values_flattened):
                        value = data_array[i : i + len(values_flattened)]  # noqa: E203
                        i += len(value)
                        value = value.reshape(sh).tolist()
                        parameter_dict.update({key: value})
                    else:
                        raise ValueError(
                            'Only "_" characters (or a list of them) should be used to '
                            "indicate parameters updated from file in parallel"
                        )
            yield index, parameter_dict

    def _sequential_parameters(self) -> t.Generator[t.Tuple[int, dict], None, None]:
        """TBW.

        :param processor:
        :return:
        """
        for step in self.enabled_steps:  # type: ParameterValues
            key = step.key  # type : str
            for index, value in enumerate(step):
                parameter_dict = {key: value}
                yield index, parameter_dict

    def _embedded_indices(self) -> "t.Iterator[t.Tuple]":
        step_ranges = []
        for step in self.enabled_steps:
            step_ranges.append(range(len(step)))
        return itertools.product(*step_ranges)

    def _embedded_parameters(
        self,
    ) -> t.Generator[t.Tuple[t.Tuple, t.Dict[str, t.Any]], None, None]:
        """TBW.

        :param processor:
        :return:
        """
        all_steps = self.enabled_steps
        keys = [step.key for step in self.enabled_steps]
        for indices, params in zip(
            self._embedded_indices(), itertools.product(*all_steps)
        ):
            parameters = {}
            for key, value in zip(keys, params):
                parameters.update({key: value})
            yield indices, parameters

    def _parameter_it(self) -> t.Callable:
        if self.parametric_mode == ParametricMode.Embedded:
            return self._embedded_parameters

        elif self.parametric_mode == ParametricMode.Sequential:
            return self._sequential_parameters

        elif self.parametric_mode == ParametricMode.Parallel:
            return self._parallel_parameters
        else:
            raise NotImplementedError

    def _processors_it(
        self, processor: "Processor"
    ) -> t.Generator[
        t.Tuple["Processor", t.Union[int, t.Tuple[int]], t.Dict], None, None
    ]:

        parameter_it = self._parameter_it()

        for index, parameter_dict in parameter_it():
            new_processor = create_new_processor(
                processor=processor, parameter_dict=parameter_dict
            )
            yield new_processor, index, parameter_dict

    def _delayed_processors(self, processor: "Processor") -> t.List:

        processors = []
        delayed_processor = delayed(processor)
        parameter_it = self._parameter_it()

        for index, parameter_dict in parameter_it():
            delayed_parameter_dict = delayed(parameter_dict)
            processors.append(
                delayed(create_new_processor)(
                    processor=delayed_processor, parameter_dict=delayed_parameter_dict
                )
            )

        return processors

    def _get_parameter_types(self) -> dict:
        for step in self.enabled_steps:
            self.parameter_types.update({step.key: step.type})
        return self.parameter_types

    def debug_mode(
        self, processor: "Processor"
    ) -> t.Tuple[t.List["Processor"], xr.Dataset]:
        processors = []
        logs = []  # type: t.List
        processor_id = 0
        for proc, index, parameter_dict in tqdm(self._processors_it(processor)):
            log = log_parameters(
                processor_id=processor_id, parameter_dict=parameter_dict
            )
            logs.append(log)
            result_proc = proc.run_pipeline()
            processors.append(result_proc)
            processor_id += 1
        final_logs = xr.combine_by_coords(logs)
        return processors, final_logs

    def _check_steps(self, processor: "Processor") -> None:

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

    def run_parametric(self, processor: "Processor") -> Result:
        """TBW."""

        self._check_steps(processor)

        types = self._get_parameter_types()

        if self.with_dask:

            raise NotImplementedError("Parametric with Dask not implemented yet.")

            # delayed_processor_list = self._delayed_processors(processor)
            # result_list = []
            #
            # for proc in delayed_processor_list:
            #     result_proc = delayed(proc.run_pipeline())
            #     result_list.append(result_proc)
            #
            # out = []

        else:

            if self.parametric_mode == ParametricMode.Embedded:

                dataset_list = []
                parameters = [
                    [] for _ in range(len(self.enabled_steps))
                ]  # type: t.List[t.List[xr.Dataset]]
                logs = []

                for processor_id, (proc, indices, parameter_dict) in enumerate(
                    tqdm(self._processors_it(processor))
                ):
                    log = log_parameters(
                        processor_id=processor_id, parameter_dict=parameter_dict
                    )
                    logs.append(log)

                    result_proc = proc.run_pipeline()

                    for i, coordinate in enumerate(parameter_dict):

                        #  appending to dataset of parameters
                        parameter_ds = parameter_to_dataset(
                            parameter_dict=parameter_dict,
                            index=indices[i],
                            coordinate=coordinate,
                        )
                        parameters[i].append(parameter_ds)

                    ds = _embedded_dataset(
                        processor=result_proc,
                        parameter_dict=parameter_dict,
                        indices=indices,
                        types=types,
                    )
                    dataset_list.append(ds)

                final_parameters_list = [xr.combine_by_coords(p) for p in parameters]
                final_parameters_merged = xr.merge(final_parameters_list)
                final_logs = xr.combine_by_coords(logs)

                out = xr.combine_by_coords(dataset_list)

                return Result(
                    dataset=out, parameters=final_parameters_merged, logs=final_logs
                )

            elif self.parametric_mode == ParametricMode.Sequential:

                dataset_dict = {}  # type: dict
                parameters = [[] for _ in range(len(self.enabled_steps))]
                logs = []

                step_counter = -1

                for processor_id, (proc, index, parameter_dict) in enumerate(
                    tqdm(self._processors_it(processor))
                ):

                    log = log_parameters(
                        processor_id=processor_id, parameter_dict=parameter_dict
                    )
                    logs.append(log)

                    coordinate = str(list(parameter_dict)[0])
                    if index == 0:
                        dataset_dict.update({short(coordinate): []})
                        step_counter += 1

                    result_proc = proc.run_pipeline()

                    parameter_ds = parameter_to_dataset(
                        parameter_dict=parameter_dict,
                        index=index,
                        coordinate=coordinate,
                    )
                    parameters[step_counter].append(parameter_ds)

                    ds = _sequential_dataset(
                        processor=result_proc,
                        parameter_dict=parameter_dict,
                        index=index,
                        coordinate=coordinate,
                        types=types,
                    )
                    dataset_dict[short(coordinate)].append(ds)

                final_logs = xr.combine_by_coords(logs)
                final_datasets = {
                    key: xr.combine_by_coords(value)
                    for key, value in dataset_dict.items()
                }
                final_parameters_list = [xr.combine_by_coords(p) for p in parameters]
                final_parameters_merged = xr.merge(final_parameters_list)

                return Result(
                    dataset=final_datasets,
                    parameters=final_parameters_merged,
                    logs=final_logs,
                )

            elif self.parametric_mode == ParametricMode.Parallel:

                dataset_list = []
                logs = []

                for proc, index, parameter_dict in tqdm(self._processors_it(processor)):

                    log = log_parameters(
                        processor_id=index, parameter_dict=parameter_dict
                    )
                    logs.append(log)

                    result_proc = proc.run_pipeline()

                    ds = _parallel_dataset(processor=result_proc, index=index)
                    dataset_list.append(ds)

                final_ds = xr.combine_by_coords(dataset_list)
                final_log = xr.combine_by_coords(logs)
                final_parameters = final_log  # parameter dataset same as logs

                return Result(
                    dataset=final_ds, parameters=final_parameters, logs=final_log
                )

            else:
                raise ValueError("Parametric mode not specified.")

    def debug(self, processor: "Processor") -> list:
        """TBW."""
        result = []
        processor_generator = self._processors_it(processor=processor)
        for i, (_processor, _, _) in enumerate(processor_generator):
            values = []
            for step in self.enabled_steps:
                _, att = get_obj_att(_processor, step.key)
                value = get_value(_processor, step.key)
                values.append((att, value))
            logging.debug("%d: %r" % (i, values))
            result.append((i, values))
        return result


def create_new_processor(processor: "Processor", parameter_dict: dict) -> "Processor":

    new_processor = deepcopy(processor)

    for key in parameter_dict.keys():
        new_processor.set(key=key, value=parameter_dict[key])

    return new_processor


def _id(s: str) -> str:
    return s + "_id"


def short(s: str) -> str:
    return s.split(".")[-1]


def log_parameters(processor_id: int, parameter_dict: dict) -> xr.Dataset:
    """

    Parameters
    ----------
    processor_id
    parameter_dict

    Returns
    -------

    """
    out = xr.Dataset()
    for key, value in parameter_dict.items():
        da = xr.DataArray(value)
        da = da.assign_coords(coords={"id": processor_id})
        da = da.expand_dims(dim="id")
        out[short(key)] = da
    return out


def parameter_to_dataset(
    parameter_dict: dict, index: int, coordinate: str
) -> xr.Dataset:
    parameter_ds = xr.Dataset()
    parameter = xr.DataArray(
        parameter_dict[coordinate], coords={short(_id(coordinate)): index}
    )
    parameter = parameter.expand_dims(dim=short(_id(coordinate)))
    parameter_ds[short(coordinate)] = parameter
    return parameter_ds


def _parallel_dataset(processor: "Processor", index: int) -> xr.Dataset:

    rows, columns = (
        processor.detector.geometry.row,
        processor.detector.geometry.row,
    )
    coordinates = {"x": range(columns), "y": range(rows)}

    arrays = {
        "pixel": "detector.pixel.array",
        "signal": "detector.signal.array",
        "image": "detector.image.array",
    }

    ds = xr.Dataset()

    for key, array in arrays.items():

        da = xr.DataArray(
            operator.attrgetter(array)(processor),
            dims=["y", "x"],
            coords=coordinates,  # type: ignore
        )
        da = da.assign_coords({"id": index})
        da = da.expand_dims(dim="id")
        ds[key] = da

    return ds


def _sequential_dataset(
    processor: "Processor",
    parameter_dict: dict,
    index: int,
    coordinate: str,
    types: dict,
) -> xr.Dataset:

    rows, columns = (
        processor.detector.geometry.row,
        processor.detector.geometry.row,
    )
    coordinates = {"x": range(columns), "y": range(rows)}

    ds = xr.Dataset()

    arrays = {
        "pixel": "detector.pixel.array",
        "signal": "detector.signal.array",
        "image": "detector.image.array",
    }

    for key, array in arrays.items():

        da = xr.DataArray(
            operator.attrgetter(array)(processor),
            dims=["y", "x"],
            coords=coordinates,  # type: ignore
        )

        #  assigning the right coordinates based on type
        if types[coordinate] == ParameterType.Simple:
            da = da.assign_coords(
                coords={short(coordinate): parameter_dict[coordinate]}
            )
            da = da.expand_dims(dim=short(coordinate))

        elif types[coordinate] == ParameterType.Multi:
            da = da.assign_coords({short(_id(coordinate)): index})
            da = da.expand_dims(dim=short(_id(coordinate)))

        ds[key] = da

    return ds


def _embedded_dataset(
    processor: "Processor", parameter_dict: dict, indices: t.Tuple[int], types: dict
) -> xr.Dataset:

    rows, columns = (
        processor.detector.geometry.row,
        processor.detector.geometry.row,
    )
    coordinates = {"x": range(columns), "y": range(rows)}

    arrays = {
        "pixel": "detector.pixel.array",
        "signal": "detector.signal.array",
        "image": "detector.image.array",
    }

    ds = xr.Dataset()

    for key, array in arrays.items():

        da = xr.DataArray(
            operator.attrgetter(array)(processor),
            dims=["y", "x"],
            coords=coordinates,  # type: ignore
        )

        for i, coordinate in enumerate(parameter_dict):

            #  assigning the right coordinates based on type
            if types[coordinate] == ParameterType.Simple:
                da = da.assign_coords(
                    coords={short(coordinate): parameter_dict[coordinate]}
                )
                da = da.expand_dims(dim=short(coordinate))

            elif types[coordinate] == ParameterType.Multi:
                da = da.assign_coords({short(_id(coordinate)): indices[i]})
                da = da.expand_dims(dim=short(_id(coordinate)))

            else:
                raise NotImplementedError

        ds[key] = da

    return ds

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



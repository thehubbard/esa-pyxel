#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Parametric mode class and helper functions."""

import itertools
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union

import numpy as np

import pyxel
from pyxel import options_wrapper
from pyxel.exposure import Readout, run_pipeline
from pyxel.observation import (
    ParameterMode,
    ParametersType,
    ParameterType,
    ParameterValues,
    _get_short_name_with_model,
    build_datatree,
    create_new_processor,
    short,
)
from pyxel.pipelines import ResultId, get_result_id

if TYPE_CHECKING:
    import pandas as pd

    # Import 'DataTree'
    try:
        from xarray.core.datatree import DataTree
    except ImportError:
        from datatree import DataTree  # type: ignore[assignment]

    from pyxel.outputs import ObservationOutputs
    from pyxel.pipelines import Processor


@dataclass(frozen=True)
class ParameterItem:
    """Internal Parameter Item."""

    # TODO: Merge 'index' and 'parameters'
    index: tuple[int, ...]
    parameters: ParametersType
    run_index: int


@dataclass(frozen=True)
class CustomParameterItem:
    """Internal Parameter Item."""

    # TODO: Merge 'index' and 'parameters'
    index: int
    parameters: ParametersType
    run_index: int


# TODO: Add unit tests
def _get_short_dimension_names_new(
    types: Mapping[str, ParameterType],
) -> Mapping[str, str]:
    # Create potential names for the dimensions
    potential_dim_names: dict[str, str] = {}
    for param_name in types:
        short_name: str = short(param_name)

        potential_dim_names[param_name] = short_name

    # Find possible duplicates
    count_dim_names: Mapping[str, int] = Counter(potential_dim_names.values())

    duplicate_dim_names: Sequence[str] = [
        name for name, freq in count_dim_names.items() if freq > 1
    ]

    if duplicate_dim_names:
        dim_names: dict[str, str] = {}
        for param_name in types:
            short_name = potential_dim_names[param_name]

            if short_name in duplicate_dim_names:
                new_short_name: str = _get_short_name_with_model(param_name)
                dim_names[param_name] = new_short_name

            else:
                dim_names[param_name] = short_name

        return dim_names

    return potential_dim_names


# TODO: Replace this function by 'xr.merge'
# TODO: or 'datatree.merge' when it will be possible
def merge(*objects: Iterable["DataTree"]) -> "DataTree":
    """Merge any number of DataTree into a single DataTree."""
    # Import 'datatree'
    try:
        from xarray.core import datatree
    except ImportError:
        import datatree  # type: ignore[no-redef]

    import xarray as xr

    def _merge_dataset(*args: xr.Dataset) -> xr.Dataset:
        return xr.merge(args)

    _merge_datatree: Callable[..., "DataTree"] = datatree.map_over_subtree(
        _merge_dataset
    )

    return _merge_datatree(*objects)


class Observation:
    """Observation class."""

    def __init__(
        self,
        parameters: Sequence[ParameterValues],
        outputs: Optional["ObservationOutputs"] = None,
        readout: Optional[Readout] = None,
        mode: Literal["product", "sequential", "custom"] = "product",
        from_file: Optional[str] = None,  # Note: Only For 'custom' mode
        column_range: Optional[tuple[int, int]] = None,  # Note: Only For 'custom' mode
        with_dask: bool = False,
        result_type: str = "all",
        pipeline_seed: Optional[int] = None,
        working_directory: Optional[str] = None,
    ):
        self.outputs: Optional["ObservationOutputs"] = outputs
        self.readout: Readout = readout or Readout()
        self.parameter_mode: ParameterMode = ParameterMode(mode)
        self.working_directory: Optional[Path] = (
            Path(working_directory) if working_directory else None
        )
        self._parameters: Sequence[ParameterValues] = parameters

        # Set 'working_directory'
        pyxel.set_options(working_directory=self.working_directory)

        # Specific to mode 'custom'
        self._custom_file: Optional[str] = from_file
        self._custom_data: Optional["pd.DataFrame"] = None
        self._custom_columns: Optional[slice] = (
            slice(*column_range) if column_range else None
        )

        self.with_dask = with_dask
        self.parameter_types: dict[str, ParameterType] = {}
        self._result_type: ResultId = get_result_id(result_type)
        self._pipeline_seed = pipeline_seed

        if self.parameter_mode == ParameterMode.Custom:
            self._load_custom_parameters()

    def __repr__(self):
        cls_name: str = self.__class__.__name__
        return f"{cls_name}<mode={self.parameter_mode!s}, num_parameters={len(self._parameters)}>"

    @property
    def result_type(self) -> ResultId:
        """TBW."""
        return self._result_type

    @result_type.setter
    def result_type(self, value: ResultId) -> None:
        """TBW."""
        self._result_type = get_result_id(value)

    @property
    def pipeline_seed(self) -> Optional[int]:
        """TBW."""
        return self._pipeline_seed

    @pipeline_seed.setter
    def pipeline_seed(self, value: int) -> None:
        """TBW."""
        self._pipeline_seed = value

    @property
    def enabled_steps(self) -> Sequence[ParameterValues]:
        """Return a list of enabled ParameterValues."""
        out = [step for step in self._parameters if step.enabled]
        return out

    # TODO: is self._custom_data really needed?
    def _load_custom_parameters(self) -> None:
        """Load custom parameters from file."""
        from pyxel import load_table

        if self._custom_file is None:
            raise ValueError("File for custom parametric mode not specified!")

        # Read the file without forcing its data type
        all_data: "pd.DataFrame" = load_table(self._custom_file, dtype=None)
        filtered_data: "pd.DataFrame" = all_data.loc[:, self._custom_columns]

        # Sanity check
        num_columns = len(filtered_data.columns)
        all_values = [list(el.values) for el in self.enabled_steps]

        counter = Counter(chain.from_iterable(all_values))
        if "_" not in counter:
            raise ValueError("Missing at parameter '_'")

        num_parameters: int = counter["_"]
        if num_parameters != num_columns:
            raise ValueError(
                f"Custom data file has {num_columns} column(s). "
                f"{num_parameters} is/are expected ! "
            )

        self._custom_data = filtered_data

    def _custom_parameters(
        self,
    ) -> Iterator[tuple[int, ParametersType]]:
        """Generate custom mode parameters based on input file.

        Yields
        ------
        index: int
        parameter_dict: dict
        """
        # Late import to speedup start-up time
        import pandas as pd

        if not isinstance(self._custom_data, pd.DataFrame):
            raise TypeError("Custom parameters not loaded from file.")

        index: int
        row_serie: pd.Series
        for index, row_serie in self._custom_data.iterrows():
            row: Sequence[Union[Number, str]] = row_serie.to_list()

            i: int = 0
            parameter_dict: ParametersType = {}
            for step in self.enabled_steps:
                key: str = step.key

                # TODO: this is confusing code. Fix this.
                #       Furthermore 'step.values' should be a `List[int, float]` and not a `str`
                if step.values == "_":
                    parameter_dict[key] = row[i]

                elif isinstance(step.values, Sequence):
                    values: np.ndarray = np.array(step.values)
                    values_flattened = values.flatten()

                    # TODO: find a way to remove the ignore
                    if not all(x == "_" for x in values_flattened):
                        raise ValueError(
                            'Only "_" characters (or a list of them) should be used to '
                            "indicate parameters updated from file in custom mode"
                        )

                    value: Sequence[Union[Number, str]] = row[
                        i : i + len(values_flattened)
                    ]
                    assert len(value) == len(step.values)

                    parameter_dict[key] = value

                else:
                    raise NotImplementedError

                i += len(step.values)

            yield index, parameter_dict

    def _sequential_parameters(self) -> Iterator[tuple[int, ParametersType]]:
        """Generate sequential mode parameters.

        Yields
        ------
        index: int
        parameter_dict: dict
        """
        index = 0

        step: ParameterValues
        for step in self.enabled_steps:
            key: str = step.key
            for value in step:
                parameter_dict: ParametersType = {key: value}
                yield index, parameter_dict

                index += 1

    def _product_indices(self) -> Iterator[tuple]:
        """Return an iterator of product parameter indices.

        Returns
        -------
        iterator
        """
        step_ranges = [range(len(step)) for step in self.enabled_steps]
        out = itertools.product(*step_ranges)
        return out

    def _product_parameters(
        self,
    ) -> Iterator[tuple[tuple, dict[str, Any]]]:
        """Generate product mode parameters.

        Yields
        ------
        indices: tuple
        parameter_dict: dict
        """
        all_steps = self.enabled_steps
        keys = [step.key for step in self.enabled_steps]
        for indices, params in zip(
            self._product_indices(), itertools.product(*all_steps)
        ):
            parameter_dict = {}
            for key, value in zip(keys, params):
                parameter_dict.update({key: value})
            yield indices, parameter_dict

    def _get_parameters_item(
        self, processor: "Processor"
    ) -> Sequence[Union[ParameterItem, CustomParameterItem]]:
        # Late import to speedup start-up time
        import toolz

        if self.parameter_mode == ParameterMode.Product:
            params_it: Iterator = self._product_parameters()

            return [
                ParameterItem(index=index, parameters=parameter_dict, run_index=n)
                for n, (index, parameter_dict) in enumerate(params_it)
            ]

        elif self.parameter_mode == ParameterMode.Sequential:
            # Get default values for all unique parameters
            params_all_keys: Sequence[str] = [
                param_value.key for param_value in self.enabled_steps
            ]
            params_unique_keys: Iterator[str] = toolz.unique(params_all_keys)

            params_defaults: ParametersType = {
                key: processor.get(key) for key in params_unique_keys
            }

            params_it = self._sequential_parameters()

            return [
                CustomParameterItem(
                    index=index,
                    # parameters=params_defaults | parameter_dict,
                    parameters={**params_defaults, **parameter_dict},
                    run_index=n,
                )
                for n, (index, parameter_dict) in enumerate(params_it)
            ]

        elif self.parameter_mode == ParameterMode.Custom:
            params_it = self._custom_parameters()

            return [
                CustomParameterItem(index=index, parameters=parameter_dict, run_index=n)
                for n, (index, parameter_dict) in enumerate(params_it)
            ]

        else:
            raise ValueError("Parametric mode not specified.")

    def _get_parameter_types(self) -> Mapping[str, ParameterType]:
        """Check for each step if parameters can be used as dataset coordinates (1D, simple) or not (multi)."""
        for step in self.enabled_steps:
            self.parameter_types.update({step.key: step.type})
        return self.parameter_types

    def validate_steps(self, processor: "Processor") -> None:
        """Validate enabled parameter steps in processor before running the pipelines.

        Parameters
        ----------
        processor: Processor

        Raises
        ------
        KeyError
            If a 'step' is missing in the configuration.
        ValueError
            If a model referenced in the configuration has not been enabled.
        """
        step: ParameterValues
        for step in self.enabled_steps:
            key: str = step.key
            if not processor.has(key):
                raise KeyError(f"Missing parameter: {key!r} in steps.")

            # TODO: the string literal expressions are difficult to maintain.
            #     Example: 'pipeline.', '.arguments', '.enabled'
            #     We may want to consider an API for this.
            # Proposed API:
            # value = operator.attrgetter(step.key)(processor)
            if "pipeline." in key:
                model_name: str = key[: key.find(".arguments")]
                model_enabled: str = model_name + ".enabled"
                if not processor.get(model_enabled):
                    raise ValueError(
                        f"The '{model_name}' model referenced in Observation"
                        " configuration has not been enabled in yaml config!"
                    )

            if (
                any(x == "_" for x in step.values[:])
                and self.parameter_mode != ParameterMode.Custom
            ):
                raise ValueError(
                    "Either define 'custom' as parametric mode or "
                    "do not use '_' character in 'values' field"
                )

    def run_observation_without_datatree(self, processor: "Processor") -> None:
        """Run the observation pipelines."""
        # Late import to speedup start-up time
        import dask.bag as db
        from tqdm.auto import tqdm

        # validation
        self.validate_steps(processor)

        types: Mapping[str, ParameterType] = self._get_parameter_types()
        dim_names: Mapping[str, str] = _get_short_dimension_names_new(types)

        parameters: Sequence[Union[ParameterItem, CustomParameterItem]] = (
            self._get_parameters_item(processor=processor)
        )

        if self.with_dask:
            datatree_bag: db.Bag = db.from_sequence(parameters).map(
                options_wrapper(working_directory=self.working_directory)(
                    self._apply_exposure_pipeline_without_datatree
                ),
                dimension_names=dim_names,
                processor=processor,
                types=types,
            )

            _ = datatree_bag.compute()
        else:
            for el in tqdm(parameters):
                self._apply_exposure_pipeline_without_datatree(
                    el,
                    dimension_names=dim_names,
                    processor=processor,
                    types=types,
                )

    def _run_observation_datatree(
        self,
        processor: "Processor",
        with_inherited_coords: bool,
    ) -> "DataTree":
        """Run the observation pipelines and return a `DataTree` object."""
        # Late import to speedup start-up time
        from tqdm.auto import tqdm

        # Validate the processor steps before running the pipeline
        self.validate_steps(processor)

        # Retrieve the types of parameters and assign short dimension names
        types: Mapping[str, ParameterType] = self._get_parameter_types()
        dim_names: Mapping[str, str] = _get_short_dimension_names_new(types)

        if self.with_dask:
            final_datatree = build_datatree(
                dim_names=dim_names,
                parameter_mode=self.parameter_mode,
                enabled_steps=self.enabled_steps,
                processor=processor,
                with_inherited_coords=with_inherited_coords,
                readout=self.readout,
                result_type=self.result_type,
                pipeline_seed=self.pipeline_seed,
            )

        else:
            # TODO: Create new class for 'Sequence[Union[ParameterItem, CustomParameterItem]]'
            # Fetch the observation parameters to be passed to the pipeline
            parameters: Sequence[Union[ParameterItem, CustomParameterItem]] = (
                self._get_parameters_item(processor=processor)
            )

            # If Dask is not enabled, process each parameter sequentially
            datatree_list: Iterator["DataTree"] = (
                self._apply_exposure_pipeline(
                    el,
                    dimension_names=dim_names,
                    processor=processor,
                    types=types,
                    with_inherited_coords=with_inherited_coords,
                )
                for el in tqdm(parameters)
            )

            # Merge the sequentially processed DataTrees into the final result
            final_datatree = merge(*datatree_list)

        # Assign the running mode to the final DataTree attributes
        parameter_name: str = self.parameter_mode.name
        final_datatree.attrs["running mode"] = f"Observation - {parameter_name}"

        # See issue #276. TODO: Is this still valid ?
        # if not isinstance(final_dataset, xr.Dataset):
        #     raise TypeError("Expecting 'Dataset'.")

        return final_datatree

    def _apply_exposure_pipeline_without_datatree(
        self,
        param_item: Union[ParameterItem, CustomParameterItem],
        dimension_names: Mapping[str, str],
        processor: "Processor",
        types: Mapping[str, ParameterType],
    ) -> None:
        new_processor = create_new_processor(
            processor=processor,
            parameter_dict=param_item.parameters,
        )

        # run the pipeline
        _ = run_pipeline(
            processor=new_processor,
            readout=self.readout,
            result_type=self.result_type,
            pipeline_seed=self.pipeline_seed,
            debug=False,  # Not supported in Observation mode
            with_inherited_coords=False,
        )

        if self.outputs:
            _ = self.outputs.save_to_file(
                processor=new_processor,
                run_number=param_item.run_index,
            )

    def _apply_exposure_pipeline(
        self,
        param_item: Union[ParameterItem, CustomParameterItem],
        dimension_names: Mapping[str, str],
        processor: "Processor",
        types: Mapping[str, ParameterType],
        with_inherited_coords: bool,
        with_outputs: bool = True,  # TODO: Refactor this
        with_extra_dims: bool = True,  # TODO: Refactor this
    ) -> "DataTree":
        """Run a single exposure pipeline for a given parameter item."""
        # Create a new processor using the given parameters
        new_processor = create_new_processor(
            processor=processor,
            parameter_dict=param_item.parameters,
        )

        # Run a single pipeline for the given parameters
        try:
            data_tree: "DataTree" = run_pipeline(
                processor=new_processor,
                readout=self.readout,
                result_type=self.result_type,
                pipeline_seed=self.pipeline_seed,
                debug=False,  # Not supported in Observation mode
                with_inherited_coords=with_inherited_coords,
            )
        except Exception as exc:
            # In Python 3.11+, add context notes to the exception
            if sys.version_info >= (3, 11):
                exc.add_note(
                    "This error occurred in 'Observation' mode with the following parameters:"
                )

                for key, value in param_item.parameters.items():
                    exc.add_note(f"  - {key!r}: {value!r}")

            raise

        # Save the outputs if configured
        if with_outputs and self.outputs:
            _ = self.outputs.save_to_file(
                processor=new_processor,
                run_number=param_item.run_index,
            )

        if not with_extra_dims:
            return data_tree

        else:
            # Can also be done outside dask in a loop
            if isinstance(param_item, ParameterItem):
                final_data_tree = _add_product_parameters(
                    data_tree=data_tree,
                    parameter_dict=param_item.parameters,
                    indexes=param_item.index,
                    dimension_names=dimension_names,
                    types=types,
                )

            else:
                final_data_tree = _add_custom_parameters(
                    data_tree=data_tree,
                    parameter_dict=param_item.parameters,
                    index=param_item.index,
                    dimension_names=dimension_names,
                    types=types,
                )

            return final_data_tree


def _add_custom_parameters(
    data_tree: "DataTree",
    parameter_dict: ParametersType,
    index: int,
    dimension_names: Mapping[str, str],
    types: Mapping[str, ParameterType],
) -> "DataTree":
    """Add coordinate "index" to the dataset.

    Parameters
    ----------
    data_tree: Dataset
    index: int

    Returns
    -------
    DataTree
    """
    # Late import to speedup start-up time
    import pandas as pd
    import xarray as xr

    data_tree = data_tree.expand_dims({"id": [index]})

    for coordinate_name, param_value in parameter_dict.items():
        short_name: str = dimension_names[coordinate_name]

        #  assigning the right coordinates based on type
        if types[coordinate_name] == ParameterType.Simple:
            data_tree = data_tree.assign_coords(
                {short_name: ("id", pd.Index([param_value]))}
            )

        elif types[coordinate_name] == ParameterType.Multi:
            data = np.array(param_value)
            data_array = xr.DataArray(data).expand_dims({"id": [index]})
            data_tree = data_tree.assign_coords({short_name: data_array})

        else:
            raise NotImplementedError

    return data_tree


def _add_product_parameters(
    data_tree: "DataTree",
    parameter_dict: ParametersType,
    indexes: tuple[int, ...],
    dimension_names: Mapping[str, str],
    types: Mapping[str, ParameterType],
) -> "DataTree":
    """Add true coordinates or index to product mode dataset.

    Parameters
    ----------
    data_tree : DataTree
    parameter_dict : dict
    types : dict

    Returns
    -------
    DataTree
    """
    import xarray as xr

    dim_idx = 0

    # TODO: join 'indexes' and 'parameter_dict'
    for index, (coordinate_name, param_value) in zip(indexes, parameter_dict.items()):
        short_name: str = dimension_names[coordinate_name]

        #  assigning the right coordinates based on type
        if types[coordinate_name] == ParameterType.Simple:
            data_tree = data_tree.expand_dims(dim={short_name: [param_value]})

        elif types[coordinate_name] == ParameterType.Multi:
            data = np.array(param_value)

            if data.ndim == 1:
                data_array = xr.DataArray(
                    data,
                    dims=f"dim_{dim_idx}",
                    coords={f"dim_{dim_idx}": range(len(data))},
                ).expand_dims(dim={f"{short_name}_id": [index]})

                dim_idx += 1

            elif data.ndim == 2:
                shape_0, shape_1 = data.shape
                data_array = xr.DataArray(
                    data,
                    dims=[f"dim_{dim_idx}", f"dim_{dim_idx + 1}"],
                    coords={
                        f"dim_{dim_idx}": range(shape_0),
                        f"dim_{dim_idx + 1}": range(shape_1),
                    },
                ).expand_dims(dim={f"{short_name}_id": [index]})

                dim_idx += 2

            else:
                raise NotImplementedError

            data_tree = data_tree.expand_dims(
                {f"{short_name}_id": [index]}
            ).assign_coords({short_name: data_array})

        else:
            raise NotImplementedError

    return data_tree

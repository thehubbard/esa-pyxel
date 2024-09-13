#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Subpackage for running Observation mode with Dask enabled."""

from collections.abc import Hashable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from pyxel.exposure import Readout, run_pipeline
from pyxel.observation import CustomMode, ProductMode, SequentialMode
from pyxel.pipelines import ResultId

if TYPE_CHECKING:
    import xarray as xr

    # Import 'DataTree'
    try:
        from xarray.core.datatree import DataTree
    except ImportError:
        from datatree import DataTree  # type: ignore[assignment]

    from pyxel.pipelines import Processor


@dataclass
class VariableMetadata:
    """Store metadata for a Xarray data variable.

    Parameters
    ----------
    dims
        The dimensions of the data variable.
    attrs
        The attributes associated with the data variable.
    dtype
        The data type of the data variable.
    """

    dims: tuple[Hashable, ...]
    attrs: Mapping[str, Any]
    dtype: np.dtype


@dataclass
class CoordinateMetadata:
    """Store metadata for a coordinate in an Xarray Dataset.

    Parameters
    ----------
    data
        The data associated with the coordinates
    attrs
        The attributes of the coordinates
    """

    data: np.ndarray
    attrs: Mapping[str, Any]


@dataclass
class DatasetMetadata:
    """Store metadata for a Xarray dataset.

    It includes its dimensions, data variables, coordinates and attributes.

    Parameters
    ----------
    dims
        The dimension(s) of the dataset.
    data_vars
        Metadata for the data variables within the dataset.
    coords
        Metadata for the coordinates within the dataset.
    attrs
        Attributes associates with dataset.
    """

    dims: tuple[Hashable, ...]
    data_vars: Mapping[str, VariableMetadata]
    coords: Mapping[Hashable, CoordinateMetadata]
    attrs: Mapping[Hashable, Any]

    def to_coords(self) -> Mapping[Hashable, "xr.DataArray"]:
        # Late import to speedup start-up time
        import xarray as xr

        dct: dict[Hashable, xr.DataArray] = {}

        meta_coord: CoordinateMetadata
        for coord_name, meta_coord in self.coords.items():
            dct[coord_name] = xr.DataArray(
                meta_coord.data,
                dims=[coord_name],
                attrs=meta_coord.attrs,
            )

        return dct


def build_metadata(data_tree: "DataTree") -> Mapping[str, DatasetMetadata]:
    metadata = {}

    all_paths: Sequence[str] = sorted(data_tree.groups)

    for path in all_paths:
        sub_data_tree: Union["DataTree", xr.DataArray] = data_tree[path]

        dims: tuple[Hashable, ...] = tuple(sub_data_tree.dims)

        metadata[path] = DatasetMetadata(
            dims=dims,
            data_vars={
                key: VariableMetadata(
                    dims=value.dims,
                    attrs=value.attrs,
                    dtype=value.dtype,
                )
                for key, value in sub_data_tree.data_vars.items()
            },
            coords={
                key: CoordinateMetadata(attrs=value.attrs, data=value.data)
                for key, value in sub_data_tree.coords.items()
            },
            attrs=sub_data_tree.attrs,
        )

    return metadata


def _get_output_core_dimensions(
    all_metadata: Mapping[str, DatasetMetadata],
) -> Sequence[tuple]:
    lst = []

    metadata: DatasetMetadata
    for metadata in all_metadata.values():
        if not metadata.data_vars:
            continue

        data_variable: VariableMetadata
        for data_variable in metadata.data_vars.values():
            lst.append(data_variable.dims)  # noqa: PERF401

    return lst


def _get_output_dtypes(
    all_metadata: Mapping[str, DatasetMetadata],
) -> Sequence[np.dtype]:
    lst = []

    metadata: DatasetMetadata
    for metadata in all_metadata.values():
        if not metadata.data_vars:
            continue

        data_variable: VariableMetadata
        for data_variable in metadata.data_vars.values():
            lst.append(data_variable.dtype)  # noqa: PERF401

    return lst


def _get_output_sizes(
    all_metadata: Mapping[str, DatasetMetadata],
) -> Mapping[Hashable, int]:
    result: dict[Hashable, int] = {}

    metadata: DatasetMetadata
    for metadata in all_metadata.values():
        if not metadata.data_vars:
            continue

        coord_sizes: Mapping[Hashable, int] = {
            coord_key: len(meta_coords.data)
            for coord_key, meta_coords in metadata.coords.items()
        }

        for coord_key in coord_sizes:
            if coord_key in result:
                raise NotImplementedError

        result.update(coord_sizes)

    return result


def _apply_exposure_from_tuple(
    params_tuple: tuple,
    dimension_names: Mapping[str, str],
    processor: "Processor",
    with_inherited_coords: bool,
    readout: Readout,
    result_type: ResultId,
    pipeline_seed: Optional[int],
) -> "DataTree":
    # TODO: Fix this
    if with_inherited_coords is False:
        raise NotImplementedError

    if len(dimension_names) != len(params_tuple):
        raise NotImplementedError

    dct = dict(zip(dimension_names, params_tuple))
    new_processor: Processor = processor.replace(dct)

    data_tree: "DataTree" = run_pipeline(
        processor=new_processor,
        readout=readout,
        result_type=result_type,
        pipeline_seed=pipeline_seed,
        debug=False,  # Not supported in Observation mode
        with_inherited_coords=with_inherited_coords,
    )

    return data_tree


def _build_metadata(
    params_tuple: tuple,
    dimension_names: Mapping[str, str],
    processor: "Processor",
    with_inherited_coords: bool,
    readout: Readout,
    result_type: ResultId,
    pipeline_seed: Optional[int],
) -> Mapping[str, DatasetMetadata]:
    data_tree: "DataTree" = _apply_exposure_from_tuple(
        params_tuple=params_tuple,
        dimension_names=dimension_names,
        processor=processor,
        with_inherited_coords=with_inherited_coords,
        readout=readout,
        result_type=result_type,
        pipeline_seed=pipeline_seed,
    )

    return build_metadata(data_tree)


def _apply_exposure_tuple_to_array(
    params_tuple: tuple,
    # idx_1d: np.ndarray,
    # run_index: int,
    dimension_names: Mapping[str, str],
    all_metadata: Mapping[str, DatasetMetadata],
    processor: "Processor",
    # types: Mapping[str, ParameterType],
    with_inherited_coords: bool,
    readout: Readout,
    result_type: ResultId,
    pipeline_seed: Optional[int],
) -> tuple[np.ndarray, ...]:
    data_tree: "DataTree" = _apply_exposure_from_tuple(
        params_tuple=params_tuple,
        dimension_names=dimension_names,
        processor=processor,
        with_inherited_coords=with_inherited_coords,
        readout=readout,
        result_type=result_type,
        pipeline_seed=pipeline_seed,
    )

    # Convert the result from 'DataTree' to a tuple of numpy array(s)
    output_data: list[np.ndarray] = []

    metadata: DatasetMetadata
    for path, metadata in all_metadata.items():
        for key in metadata.data_vars:
            output_data.append(data_tree[f"{path}/{key}"].to_numpy())  # noqa: PERF401

    return tuple(output_data)


def build_datatree(
    dim_names: Mapping[str, str],
    parameter_mode: Union[ProductMode, SequentialMode, CustomMode],
    processor: "Processor",
    with_inherited_coords: bool,
    readout: Readout,
    result_type: ResultId,
    pipeline_seed: Optional[int],
) -> "DataTree":
    # Late import to speedup start-up time
    import xarray as xr

    # Import 'DataTree'
    try:
        from xarray.core.datatree import DataTree
    except ImportError:
        from datatree import DataTree  # type: ignore[assignment]

    # Get parameters as a DataArray
    params_dataarray: xr.DataArray = parameter_mode.create_params(dim_names=dim_names)

    # Get the first parameter
    first_param: tuple = (
        params_dataarray.head(1)  # Get the first parameter(s)
        .squeeze()  # Remove dimensions of length 1
        .to_numpy()  # Convert to a numpy array
        .tolist()  # Convert to a tuple
    )

    # Extract metadata
    all_metadata: Mapping[str, DatasetMetadata] = _build_metadata(
        params_tuple=first_param,
        dimension_names=dim_names,
        processor=processor,
        with_inherited_coords=with_inherited_coords,
        readout=readout,
        result_type=result_type,
        pipeline_seed=pipeline_seed,
    )

    # Get the output core dimensions
    output_core_dims: Sequence[tuple] = _get_output_core_dimensions(all_metadata)

    # Get output sizes
    output_sizes: Mapping[Hashable, int] = _get_output_sizes(all_metadata)

    # Get output dtypes
    output_dtypes: Sequence[np.dtype] = _get_output_dtypes(all_metadata)

    # Create 'Dask' data arrays
    dask_dataarrays: tuple[xr.DataArray, ...] = xr.apply_ufunc(
        _apply_exposure_tuple_to_array,  # Function to apply
        params_dataarray.chunk(1),  # Argument 'params_tuple'
        kwargs={  # other arguments
            "dimension_names": dim_names,
            "processor": processor,
            "all_metadata": all_metadata,
            # "types": types,
            "with_inherited_coords": with_inherited_coords,
            "readout": readout,
            "result_type": result_type,
            "pipeline_seed": pipeline_seed,
        },
        input_core_dims=[[]],
        output_core_dims=output_core_dims,
        vectorize=True,  # loop over non-core dims
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": output_sizes},
        output_dtypes=output_dtypes,  # TODO: Move this to 'dask_gufunc_kwargs'
    )

    # Rebuild the DataTree from 'all_dataarrays'
    dct: dict[str, Union[xr.Dataset, DataTree]] = {}

    idx = 0

    path: str
    partial_metadata: DatasetMetadata
    for path, partial_metadata in all_metadata.items():
        if not partial_metadata.data_vars:
            # TODO: Use this ?
            # assert not partial_metadata.dims
            # assert not partial_metadata.coords

            empty_data_tree: DataTree = DataTree()
            empty_data_tree.attrs = dict(partial_metadata.attrs)

            dct[path] = empty_data_tree

        else:
            assert partial_metadata.dims
            assert partial_metadata.coords

            data_set = xr.Dataset(attrs=partial_metadata.attrs)

            var_name: str
            metadata_vars: VariableMetadata
            for var_name, metadata_vars in partial_metadata.data_vars.items():
                data_set[var_name] = (
                    dask_dataarrays[idx]
                    .rename(var_name)
                    .assign_attrs(metadata_vars.attrs)
                )

                idx += 1
                assert idx <= len(dask_dataarrays)

            coords: Mapping[Hashable, xr.DataArray] = partial_metadata.to_coords()
            dct[path] = data_set.assign_coords(coords)

    # for (key, partial_metadata), data_array in zip(metadata.items(), dask_dataarrays):
    #     *_, name = key.split("/")
    #     dct[key] = data_array.rename(name).assign_attrs(partial_metadata.attrs)

    # del dct['/bucket']

    if xr.__version__ <= "2024.9.0":
        final_datatree = DataTree.from_dict(deepcopy(dct))  # type: ignore[arg-type]
    else:
        final_datatree = DataTree.from_dict(dct)  # type: ignore[arg-type]

    return final_datatree

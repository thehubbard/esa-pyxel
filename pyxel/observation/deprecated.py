#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.


import warnings
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple, Union

from pyxel.observation import (
    ParametersType,
    ParameterType,
    _get_short_name_with_model,
    short,
)

if TYPE_CHECKING:
    import xarray as xr


class ObservationResult(NamedTuple):
    """Result class for observation class."""

    dataset: Union["xr.Dataset", dict[str, "xr.Dataset"]]
    parameters: "xr.Dataset"
    logs: "xr.Dataset"


def _id(s: str) -> str:  # pragma: no cover
    """Add _id to the end of a string."""
    warnings.warn(
        "Deprecated. Will be removed in Pyxel 2.0", DeprecationWarning, stacklevel=1
    )

    out = s + "_id"
    return out


def parameter_to_dataset(
    parameter_dict: dict,
    dimension_names: Mapping[str, str],
    index: int,
    coordinate_name: str,
) -> "xr.Dataset":  # pragma: no cover
    """Return a specific parameter dataset from a parameter dictionary.

    Parameters
    ----------
    parameter_dict: dict
    dimension_names
    index: int
    coordinate_name: str

    Returns
    -------
    Dataset
    """
    warnings.warn(
        "Deprecated. Will be removed in Pyxel 2.0",
        DeprecationWarning,
        stacklevel=1,
    )

    # Late import to speedup start-up time
    import xarray as xr

    parameter_ds = xr.Dataset()
    parameter = xr.DataArray(parameter_dict[coordinate_name])

    # TODO: Dirty hack. Fix this !
    short_name: str = dimension_names[coordinate_name]

    if short_name.endswith("_id"):
        short_coord_name = short_name[:-3]
        short_coord_name_id = short_name
    else:
        short_coord_name = short_name
        short_coord_name_id = f"{short_name}_id"

    parameter = parameter.assign_coords({short_coord_name_id: index})
    parameter = parameter.expand_dims(dim=short_coord_name_id)
    parameter_ds[short_coord_name] = parameter

    return parameter_ds


def _add_custom_parameters_deprecated(
    ds: "xr.Dataset", index: int
) -> "xr.Dataset":  # pragma: no cover
    """Add coordinate "index" to the dataset.

    Parameters
    ----------
    ds: Dataset
    index: int

    Returns
    -------
    Dataset
    """
    warnings.warn(
        "Deprecated. Will be removed in Pyxel 2.0", DeprecationWarning, stacklevel=1
    )

    ds = ds.assign_coords({"id": index})
    ds = ds.expand_dims(dim="id")

    return ds


def _get_final_short_name(
    name: str, param_type: ParameterType
) -> str:  # pragma: no cover
    warnings.warn(
        "Deprecated. Will be removed in Pyxel 2.0", DeprecationWarning, stacklevel=1
    )

    if param_type == ParameterType.Simple:
        return name
    elif param_type == ParameterType.Multi:
        return _id(name)
    else:
        raise NotImplementedError


def _get_short_dimension_names(
    types: Mapping[str, ParameterType],
) -> Mapping[str, str]:  # pragma: no cover
    warnings.warn(
        "Deprecated. Will be removed in Pyxel 2.0", DeprecationWarning, stacklevel=1
    )
    # Create potential names for the dimensions
    potential_dim_names: dict[str, str] = {}
    for param_name, param_type in types.items():
        short_name: str = short(param_name)

        potential_dim_names[param_name] = _get_final_short_name(
            name=short_name, param_type=param_type
        )

    # Find possible duplicates
    count_dim_names: Mapping[str, int] = Counter(potential_dim_names.values())

    duplicate_dim_names: Sequence[str] = [
        name for name, freq in count_dim_names.items() if freq > 1
    ]

    if duplicate_dim_names:
        dim_names: dict[str, str] = {}
        for param_name, param_type in types.items():
            short_name = potential_dim_names[param_name]

            if short_name in duplicate_dim_names:
                new_short_name: str = _get_short_name_with_model(param_name)
                dim_names[param_name] = _get_final_short_name(
                    name=new_short_name, param_type=param_type
                )

            else:
                dim_names[param_name] = short_name

        return dim_names

    return potential_dim_names


def log_parameters(
    processor_id: int, parameter_dict: dict
) -> "xr.Dataset":  # pragma: no cover
    """Return parameters in the current processor in a xarray dataset.

    Parameters
    ----------
    processor_id: int
    parameter_dict: dict

    Returns
    -------
    Dataset
    """
    warnings.warn(
        "Deprecated. Will be removed in Pyxel 2.0",
        DeprecationWarning,
        stacklevel=1,
    )

    # Late import to speedup start-up time
    import xarray as xr

    out = xr.Dataset()
    for key, value in parameter_dict.items():
        da = xr.DataArray(value)
        da = da.assign_coords(coords={"id": processor_id})
        da = da.expand_dims(dim="id")
        out[short(key)] = da
    return out


def _add_sequential_parameters_deprecated(
    ds: "xr.Dataset",
    parameter_dict: ParametersType,
    dimension_names: Mapping[str, str],
    index: int,
    coordinate_name: str,
    types: Mapping[str, ParameterType],
) -> "xr.Dataset":  # pragma: no cover
    """Add true coordinates or index to sequential mode dataset.

    Parameters
    ----------
    ds : Dataset
    parameter_dict : dict
    dimension_names
    index : int
    coordinate_name : str
    types : dict

    Returns
    -------
    Dataset
    """
    warnings.warn(
        "Deprecated. Will be removed in Pyxel 2.0", DeprecationWarning, stacklevel=1
    )

    #  assigning the right coordinates based on type
    short_name: str = dimension_names[coordinate_name]

    if types[coordinate_name] == ParameterType.Simple:
        ds = ds.assign_coords(coords={short_name: parameter_dict[coordinate_name]})
        ds = ds.expand_dims(dim=short_name)

    elif types[coordinate_name] == ParameterType.Multi:
        ds = ds.assign_coords({short_name: index})
        ds = ds.expand_dims(dim=short_name)

    return ds


def _add_product_parameters_deprecated(  # pragma: no cover
    ds: "xr.Dataset",
    parameter_dict: ParametersType,
    dimension_names: Mapping[str, str],
    indices: tuple[int, ...],
    types: Mapping[str, ParameterType],
) -> "xr.Dataset":  # pragma: no cover
    """Add true coordinates or index to product mode dataset.

    Parameters
    ----------
    ds: Dataset
    parameter_dict: dict
    indices: tuple
    types: dict

    Returns
    -------
    Dataset
    """
    warnings.warn(
        "Deprecated. Will be removed in Pyxel 2.0", DeprecationWarning, stacklevel=1
    )

    # TODO: Implement for coordinate 'multi'
    for i, (coordinate_name, param_value) in enumerate(parameter_dict.items()):
        short_name: str = dimension_names[coordinate_name]

        #  assigning the right coordinates based on type
        if types[coordinate_name] == ParameterType.Simple:
            ds = ds.assign_coords(coords={short_name: param_value})
            ds = ds.expand_dims(dim=short_name)

        elif types[coordinate_name] == ParameterType.Multi:
            ds = ds.assign_coords({short_name: indices[i]})
            ds = ds.expand_dims(dim=short_name)

        else:
            raise NotImplementedError

    return ds


def to_tuples(data: Iterable) -> tuple:  # pragma: no cover
    warnings.warn(
        "Deprecated. Will be removed in Pyxel 2.0", DeprecationWarning, stacklevel=1
    )

    lst: list = []
    for el in data:
        if isinstance(el, Iterable) and not isinstance(el, str):
            lst.append(to_tuples(el))
        else:
            lst.append(el)

    return tuple(lst)


def compute_final_sequential_dataset(
    list_of_index_and_parameter: list,
    list_of_datasets: list,
    dimension_names: Mapping[str, str],
) -> dict[str, "xr.Dataset"]:  # pragma: no cover
    """Return a dictionary of result datasets where keys are different parameters.

    Parameters
    ----------
    list_of_index_and_parameter: list
    list_of_datasets: list
    dimension_names

    Returns
    -------
    dict
    """
    warnings.warn(
        "Deprecated. Will be removed in Pyxel 2.0",
        DeprecationWarning,
        stacklevel=1,
    )

    # Late import to speedup start-up time
    import xarray as xr

    final_dict: dict[str, list[xr.Dataset]] = {}

    for _, parameter_dict, n in list_of_index_and_parameter:
        coordinate = str(next(iter(parameter_dict)))
        coordinate_short: str = dimension_names[coordinate]

        if short(coordinate) not in final_dict:
            final_dict.update({coordinate_short: []})
            final_dict[coordinate_short].append(list_of_datasets[n])
        else:
            final_dict[coordinate_short].append(list_of_datasets[n])

    final_datasets: dict[str, xr.Dataset] = {}
    for key, value in final_dict.items():
        ds = xr.combine_by_coords(value)
        # see issue #276
        if not isinstance(ds, xr.Dataset):
            raise TypeError("Expecting 'Dataset'.")

        final_datasets.update({key: ds})

    return final_datasets

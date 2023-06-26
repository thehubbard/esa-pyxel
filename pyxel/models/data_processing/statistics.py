#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Simple model to compute basic statistics."""
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Union

import xarray as xr
from datatree import DataTree

from pyxel.detectors import Detector

if TYPE_CHECKING:
    from pyxel.data_structure import Image, Photon, Pixel, Signal


def compute_statistics(
    data_array: xr.DataArray,
    absolute_time: xr.DataArray,
    dimensions: Union[str, Sequence[str]] = ("x", "y"),
) -> xr.Dataset:
    """Compute basic statistics and save it into a Dataset.

    Parameters
    ----------
    data_array : DataArray
        Input array which contains the data.
    absolute_time : DataArray
        Absolute time(s) of the data.
    dimensions : str or Sequence of str
        Dimensions, where the statistics should be computed on. Default is ("x", "y").

    Returns
    -------
    Dataset
    """

    var = data_array.var(dim=dimensions)
    mean = data_array.mean(dim=dimensions)
    min_array = data_array.min(dim=dimensions)
    max_array = data_array.max(dim=dimensions)
    count = data_array.count(dim=dimensions)

    dataset = xr.Dataset()
    dataset["var"] = var
    dataset["mean"] = mean
    dataset["min"] = min_array
    dataset["max"] = max_array
    dataset["count"] = count

    dataset = dataset.expand_dims(dim="time").assign_coords(time=absolute_time)

    return dataset


def statistics(
    detector: Detector,
    data_structure: Literal["pixel", "photon", "signal", "image", "all"] = "all",
) -> None:
    """Compute basic statistics.

    Parameters
    ----------
    detector : Detector
        Pyxel Detector object.
    data_structure : Literal
        Keyword to choose data structure. Can be any from:
        ("pixel", "photon", "signal", "image", "all").
        Default is "all" and computes the statistics on "pixel", "photon", "signal" and "image".

    """
    if data_structure == "all":
        names = ["pixel", "photon", "signal", "image"]
    else:
        names = [data_structure]

    parent: str = "/statistics"
    parent_partial: str = f"{parent}/partial"

    for name in names:
        # Extract data from 'detector'
        data_bucket: Union[Pixel, Photon, Signal, Image] = getattr(detector, name)
        data_array: xr.DataArray = data_bucket.to_xarray(dtype=float)

        # Get current absolute time
        absolute_time = xr.DataArray(
            [detector.absolute_time],
            dims=["time"],
            attrs={"units": "s"},
        )

        # Get statistics data
        statistics: xr.Dataset = compute_statistics(
            data_array=data_array, absolute_time=absolute_time
        )

        key: str = f"{parent}/{name}"
        key_partial: str = f"{parent_partial}/{name}"

        try:
            _ = detector.data[key_partial]
        except KeyError:
            has_key_partial = False
        else:
            has_key_partial = True

        if not has_key_partial:
            data_tree: DataTree = DataTree(statistics)
        else:
            # Concatenate data
            previous_datatree = detector.data[key_partial]
            data_tree = previous_datatree.combine_first(statistics)  # type: ignore

        if detector.pipeline_count == (detector.num_steps - 1):
            detector.data[key] = data_tree
        else:
            detector.data[key_partial] = data_tree

    # This is the last step and there is at least two steps
    if detector.num_steps > 1 and (detector.pipeline_count == (detector.num_steps - 1)):
        detector.data[parent_partial].orphan()

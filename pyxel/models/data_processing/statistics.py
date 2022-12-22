#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Simple model to compute basic statistics."""
from typing import Literal, Sequence, Union

import xarray as xr

from pyxel.detectors import Detector


def compute_statistics(
    detector: Detector,
    data_structure: Literal["pixel", "photon", "image", "signal"] = "pixel",
    dimensions: Union[str, Sequence[str]] = ("x", "y"),
) -> None:
    """Compute basic statistics.

    Parameters
    ----------
    detector
    data_structure
    dimensions
    """
    data_2d: xr.DataArray = getattr(detector, data_structure).to_xarray()
    var = data_2d.var(dim=dimensions)
    mean = data_2d.mean(dim=dimensions)
    min_array = data_2d.min(dim=dimensions)
    max_array = data_2d.max(dim=dimensions)
    count = data_2d.count(dim=dimensions)

    dataset = xr.Dataset()
    dataset["var"] = var
    dataset["mean"] = mean
    dataset["min"] = min_array
    dataset["max"] = max_array
    dataset["count"] = count

    detector.processed_data.append(dataset)

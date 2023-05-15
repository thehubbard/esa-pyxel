#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Model to compute linear regressions."""

from typing import Literal, Optional, Union

import numpy as np
import xarray as xr

from pyxel.data_structure import Image, Photon, Pixel, Signal
from pyxel.detectors import Detector


def linear_regression(
    detector: Detector,
    data_structure: Literal["pixel", "photon", "image", "signal"] = "image",
    name: Optional[str] = None,
) -> None:
    """Compute a linear regression along 'readout_time' and store it in '.data' bucket.

    Parameters
    ----------
    detector : Detector
    data_structure : 'pixel', 'photon', 'image' or 'signal'
        Data bucket to use for the linear regression.
    name : str, optional
        Name to use for the result.

    Examples
    --------
    >>> detector = CCD(...)
    >>> linear_regression(detector=detector)

    >>> detector.data
    DataTree('None', parent=None)
    └── DataTree('linear_regression')
        └── DataTree('image')
                Dimensions:        (y: 450, x: 450)
                Coordinates:
                  * y              (y) int64 0 1 2 3 4 5 6 7 ... 442 443 444 445 446 447 448 449
                  * x              (x) int64 0 1 2 3 4 5 6 7 ... 442 443 444 445 446 447 448 449
                Data variables:
                    slope          (y, x) float64 ...
                    intercept      (y, x) float64 ...
                    r2             (y, x) float64 ...
                    slope_std      (y, x) float64 ...
                    intercept_std  (y, x) float64 ...
    """
    if detector.num_steps < 3:
        raise ValueError(f"Expecting at least 3 steps. Got: {detector.num_steps}")

    if name is None:
        name = data_structure

    # Extract data from 'detector'
    data_bucket: Union[Pixel, Photon, Image, Signal] = getattr(detector, data_structure)

    x: float = detector.absolute_time
    y: xr.DataArray = data_bucket.to_xarray()

    if detector.is_first_readout:
        detector.data[f"/linear_regression/{name}/partial/n"] = 1
        detector.data[f"/linear_regression/{name}/partial/sum_x"] = x
        detector.data[f"/linear_regression/{name}/partial/sum_y"] = y
        detector.data[f"/linear_regression/{name}/partial/sum_xy"] = x * y
        detector.data[f"/linear_regression/{name}/partial/sum_x2"] = x * x
        detector.data[f"/linear_regression/{name}/partial/sum_y2"] = y * y
    else:
        detector.data[f"/linear_regression/{name}/partial/n"] += 1  # type: ignore
        detector.data[f"/linear_regression/{name}/partial/sum_x"] += x  # type: ignore
        detector.data[f"/linear_regression/{name}/partial/sum_y"] += y
        detector.data[f"/linear_regression/{name}/partial/sum_xy"] += x * y
        detector.data[f"/linear_regression/{name}/partial/sum_x2"] += x * x  # type: ignore
        detector.data[f"/linear_regression/{name}/partial/sum_y2"] += y * y

        if detector.is_last_readout:
            n = detector.data[f"/linear_regression/{name}/partial/n"]
            sum_x = detector.data[f"/linear_regression/{name}/partial/sum_x"]
            sum_y = detector.data[f"/linear_regression/{name}/partial/sum_y"]
            sum_xy = detector.data[f"/linear_regression/{name}/partial/sum_xy"]
            sum_x2 = detector.data[f"/linear_regression/{name}/partial/sum_x2"]
            sum_y2 = detector.data[f"/linear_regression/{name}/partial/sum_y2"]

            factor_x = n * sum_x2 - sum_x * sum_x  # type: ignore
            factor_y = n * sum_y2 - sum_y * sum_y  # type: ignore

            slope = (n * sum_xy - sum_x * sum_y) / factor_x  # type: ignore
            intercept = (sum_y - slope * sum_x) / n

            r2 = slope * slope * factor_x / factor_y

            factor_var = factor_y - slope * slope * factor_x
            slope_var = factor_var / ((n - 2) * factor_x)  # type: ignore
            intercept_var = sum_x2 * factor_var / (n * (n - 2) * factor_x)  # type: ignore

            slope_std = np.sqrt(slope_var)
            intercept_std = np.sqrt(intercept_var)

            if y_unit := y.attrs.get("units"):
                x_unit = "s"
                slope.attrs["unit"] = f"{y_unit}/{x_unit}"
                intercept.attrs["unit"] = y_unit
                slope_std.attrs["unit"] = f"{y_unit}/{x_unit}"  # type: ignore
                intercept_std.attrs["unit"] = y_unit  # type: ignore

            detector.data[f"/linear_regression/{name}/slope"] = slope
            detector.data[f"/linear_regression/{name}/intercept"] = intercept
            detector.data[f"/linear_regression/{name}/r2"] = r2
            detector.data[f"/linear_regression/{name}/slope_std"] = slope_std
            detector.data[f"/linear_regression/{name}/intercept_std"] = intercept_std

            detector.data[f"/linear_regression/{name}"].attrs[
                "long_name"
            ] = f"Linear regression: {name}"

            # Remove '/linear_regression/partial
            detector.data[f"/linear_regression/{name}/partial"].orphan()

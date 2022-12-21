from typing import Optional, Union

import xarray as xr


class ProcessedData:
    """Create a `ProcessedData` container.

    Examples
    --------
    >>> obj = ProcessedData()
    >>> obj.data
    <xarray.Dataset>
    Dimensions:  ()
    Data variables:
        *empty*
    >>> obj.append(xr.DataArray(...))
    """

    def __init__(self, data: Optional[xr.Dataset] = None):

        if data is None:
            data = xr.Dataset()

        self._data: xr.Dataset = data

    @property
    def data(self) -> xr.Dataset:
        return self._data

    def append(self, other: Union[xr.Dataset, xr.DataArray]) -> None:
        self._data = xr.combine_by_coords([self._data, other])

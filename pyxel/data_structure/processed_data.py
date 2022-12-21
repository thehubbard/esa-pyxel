import xarray as xr


class ProcessedData:
    def __init__(self, data: xr.Dataset):
        self.data = data

    def append(self, other: xr.Dataset):
        self.data = xr.merge([self.data, other])

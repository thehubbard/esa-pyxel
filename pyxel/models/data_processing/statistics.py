import xarray as xr

from pyxel.detectors import Detector


def compute_statistics(
    detector: Detector,
    data_structure: str = "pixel",
    dimensions=["x", "y"],
) -> None:

    data_2d = getattr(detector, data_structure).to_xarray()
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

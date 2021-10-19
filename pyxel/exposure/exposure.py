#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
"""Observation class and functions."""

import logging
import operator
import typing as t

import xarray as xr
from tqdm.auto import tqdm

from .readout import Readout

if t.TYPE_CHECKING:
    from ..outputs import CalibrationOutputs, ExposureOutputs, ObservationOutputs
    from ..pipelines import Processor


class Exposure:
    """TBW."""

    def __init__(self, outputs: "ExposureOutputs", readout: "Readout"):
        self.outputs = outputs
        self.readout = readout

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__  # type: str
        return f"{cls_name}<outputs={self.outputs!r}>"

    def run_exposure(self, processor: "Processor") -> xr.Dataset:
        """Run a an observation pipeline.

        Parameters
        ----------
        processor

        Returns
        -------
        result: xarrray.Dataset
        """
        if self.readout._num_steps == 1:
            progressbar = False
        else:
            progressbar = True

        result, _ = run_exposure_pipeline(
            processor=processor,
            readout=self.readout,
            outputs=self.outputs,
            progressbar=progressbar,
        )
        return result


def run_exposure_pipeline(
    processor: "Processor",
    readout: "Readout",
    outputs: t.Optional[
        t.Union["CalibrationOutputs", "ObservationOutputs", "ExposureOutputs"]
    ] = None,
    progressbar: bool = False,
) -> t.Tuple[xr.Dataset, "Processor"]:
    """Run standalone exposure pipeline.

    Parameters
    ----------
    processor: Processor
    readout: Readout
    outputs: DynamicOutputs
        Sampling outputs.
    progressbar: bool
        Sets visibility of progress bar.

    Returns
    -------
    final_dataset: xr.Dataset
        Results of the pipeline in an xarray dataset.
    """
    # if isinstance(detector, CCD):
    #    dynamic.non_destructive_readout = False

    num_steps = readout._num_steps
    ndreadout = readout.non_destructive
    times_linear = readout._times_linear
    start_time = readout._start_time
    end_time = readout._times[-1]
    time_step_it = readout.time_step_it()
    times = readout.times

    detector = processor.detector

    detector.set_readout(
        num_steps=num_steps,
        ndreadout=ndreadout,
        times_linear=times_linear,
        start_time=start_time,
        end_time=end_time,
    )
    # The detector should be reset before exposure
    detector.empty()

    # prepare lists for to-be-merged datasets
    list_datasets = []

    # Dimensions set by the detectors dimensions
    rows, columns = (
        processor.detector.geometry.row,
        processor.detector.geometry.col,
    )
    # Coordinates
    coordinates = {"x": range(columns), "y": range(rows)}

    if progressbar:
        pbar = tqdm(total=num_steps)

    for i, (time, step) in enumerate(
        time_step_it
    ):  # type: t.Tuple[int, t.Tuple[float, float]]

        detector.time = time
        detector.time_step = step

        logging.info("time = %.3f s", time)

        if detector.non_destructive_readout:
            detector.empty(empty_all=False)
        else:
            detector.empty(empty_all=True)

        processor.run_pipeline()
        detector.pipeline_count = i

        if outputs and detector.read_out:
            outputs.save_to_file(processor)

        out = xr.Dataset()

        # Dataset is storing the arrays at the end of this iter
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
            # Time coordinate of this iteration
            da = da.assign_coords(coords={"readout_time": time})
            da = da.expand_dims(dim="readout_time")

            out[key] = da

        if progressbar:
            pbar.update(1)
        # Append to the list of datasets
        list_datasets.append(out)

    if progressbar:
        pbar.close()

    # Combine the datasets in the list into one xarray
    final_dataset = xr.combine_by_coords(list_datasets)  # type: xr.Dataset

    return final_dataset, processor

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

from .sampling import Sampling

if t.TYPE_CHECKING:
    from ..outputs import CalibrationOutputs, ObservationOutputs, ParametricOutputs
    from ..pipelines import Processor


class Observation:
    """TBW."""

    def __init__(self, outputs: "ObservationOutputs", sampling: "Sampling"):
        self.outputs = outputs
        self.sampling = sampling

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__  # type: str
        return f"{cls_name}<outputs={self.outputs!r}>"

    def run_observation(self, processor: "Processor") -> xr.Dataset:
        """Run a an observation pipeline.

        Parameters
        ----------
        processor

        Returns
        -------
        result: xarrray.Dataset
        """
        result, _ = run_observation(
            processor=processor,
            sampling=self.sampling,
            outputs=self.outputs,
            progressbar=True,
        )
        return result


def run_observation(
    processor: "Processor",
    sampling: "Sampling",
    outputs: t.Optional["ObservationOutputs"] = None,
    progressbar: bool = False,
) -> t.Tuple[xr.Dataset, "Processor"]:
    """Run a an observation pipeline.

    Parameters
    ----------
    processor
    sampling
    outputs
    progressbar

    Returns
    -------
    result: xr.Dataset
    """
    if sampling._num_steps == 1:
        progressbar = False

    result = observation_pipeline(
        processor=processor,
        time_step_it=sampling.time_step_it(),
        num_steps=sampling._num_steps,
        ndreadout=sampling.non_destructive_readout,
        times_linear=sampling._times_linear,
        start_time=sampling._start_time,
        end_time=sampling._times[-1],
        outputs=outputs,
        progressbar=progressbar,
    )

    return result, processor


def observation_pipeline(
    processor: "Processor",
    time_step_it: t.Iterator[t.Tuple[float, float]],
    num_steps: int,
    times_linear: bool,
    end_time: float,
    start_time: float = 0.0,
    ndreadout: bool = False,
    outputs: t.Optional[
        t.Union["CalibrationOutputs", "ParametricOutputs", "ObservationOutputs"]
    ] = None,
    progressbar: bool = False,
) -> xr.Dataset:
    """Run standalone dynamic pipeline.

    Parameters
    ----------
    processor: Processor
    time_step_it: Iterator
        Iterates over pairs of times and elapsed time steps.
    num_steps: int
        Number of times.
    ndreadout: bool
        Set non destructive readout mode.
    times_linear: bool
        Set if times are linear.
    start_time: float
        Starting time.
    end_time:
        Last time.
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

    detector = processor.detector

    detector.set_sampling(
        num_steps=num_steps,
        ndreadout=ndreadout,
        times_linear=times_linear,
        start_time=start_time,
        end_time=end_time,
    )
    # The detector should be reset before exposure
    detector.reset(reset_all=True)

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
            detector.reset(reset_all=False)
        else:
            detector.reset(reset_all=True)

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

    return final_dataset

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
"""TBW."""

import logging
import operator
import typing as t

import numpy as np
import xarray as xr
from tqdm.notebook import tqdm

from pyxel.evaluator import eval_range
from pyxel.inputs_outputs import load_table

if t.TYPE_CHECKING:
    from ..inputs_outputs import DynamicOutputs
    from ..pipelines import Processor


# class DynamicResult(t.NamedTuple):
#     """Result class for parametric class."""
#
#     dataset: t.Union[xr.Dataset, t.Dict[str, xr.Dataset]]
#     # parameters: xr.Dataset
#     # logs: xr.Dataset


class Dynamic:
    """TBW."""

    def __init__(
        self,
        outputs: "DynamicOutputs",
        times: t.Optional[t.Union[t.Sequence, str]] = None,
        times_from_file: t.Optional[str] = None,
        start_time: float = 0.0,
        non_destructive_readout: bool = False,
    ):
        self.outputs = outputs

        if times is not None and times_from_file is not None:
            raise ValueError("Both times and times_from_file specified. Choose one.")
        elif times_from_file:
            self._times = load_table(times_from_file).to_numpy(
                dtype=float
            )
        elif times:
            self._times = np.array(eval_range(times), dtype=float)
        else:
            raise ValueError("Dynamic times not specified.")

        self._non_destructive_readout = non_destructive_readout

        if np.ndim(self.times) != 1:
            raise ValueError("Number of dimensions in the times array is not 1.")

        self._linear = True  # type: bool
        self._start_time = start_time  # type:float
        self._steps = np.array([])  # type: np.ndarray
        self._num_steps = 0  # type: int

        self.set_steps()

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__  # type: str
        return f"{cls_name}<outputs={self.outputs!r}>"

    @property
    def times(self):
        """TBW."""
        return self._times

    @times.setter
    def times(self, value):
        """TBW."""
        self._times = value

    @property
    def steps(self):
        """TBW."""
        return self._steps

    @steps.setter
    def steps(self, value: np.ndarray) -> None:
        """TBW."""
        self._steps = value

    def set_steps(self):
        """TBW."""
        if self._start_time == self.times[0]:
            self.steps = np.diff(self.times, axis=0)
            self.times = self.times[1:]
        else:
            self.steps = np.diff(
                np.concatenate((np.array([self._start_time]), self.times), axis=0),
                axis=0,
            )

        self._linear = np.all(self.steps == self.steps[0])

        self._num_steps = len(self.times)

    def time_it(self):
        """

        Returns
        -------

        """
        return zip(self.times, self.steps)

    @property
    def non_destructive_readout(self):
        """TBW."""
        return self._non_destructive_readout

    @non_destructive_readout.setter
    def non_destructive_readout(self, non_destructive_readout: bool) -> None:
        """TBW."""
        self._non_destructive_readout = non_destructive_readout

    def run_dynamic(self, processor: "Processor") -> xr.Dataset:
        """TBW."""
        # if isinstance(detector, CCD):
        #    dynamic.non_destructive_readout = False

        detector = processor.detector

        detector.set_dynamic(
            num_steps=self._num_steps,
            ndreadout=self._non_destructive_readout,
            linear=self._linear,
            start_time=self._start_time,
            end_time=self.times[-1],
        )
        # The detector should be reset before exposure
        detector.reset(reset_all=True)

        # prepare lists for to-be-merged datasets
        list_datasets = []

        # Dimensions set by the detectors dimensions
        rows, columns = (
            processor.detector.geometry.row,
            processor.detector.geometry.row,
        )
        # Coordinates
        coordinates = {"x": range(columns), "y": range(rows)}

        pbar = tqdm(total=self._num_steps)

        for i, (time, step) in enumerate(self.time_it()):

            detector.time = time
            detector.time_step = step

            logging.info("time = %.3f s", time)

            if detector.is_non_destructive_readout:
                detector.reset(reset_all=False)
            else:
                detector.reset(reset_all=True)

            processor.run_pipeline()
            detector.pipeline_count = i

            if detector.read_out:
                self.outputs.save_to_file(processor)

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
                da = da.assign_coords(coords={"t": processor.detector.time})
                da = da.expand_dims(dim="t")

                out[key] = da

            pbar.update(1)
            # Append to the list of datasets
            list_datasets.append(out)

        pbar.close()

        # Combine the datasets in the list into one xarray
        final_dataset = xr.combine_by_coords(list_datasets)  # type: xr.Dataset

        return final_dataset

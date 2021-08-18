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

import xarray as xr
from tqdm.notebook import tqdm
from pyxel.evaluator import eval_range
#from pyxel.inputs_outputs import load_table
import numpy as np

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
        times: t.Optional[t.Union[t.Sequence, str]],
        times_from_file: t.Optional[str],
        non_destructive_readout: bool = False,
    ):
        self.outputs = outputs
        if times is None and times_from_file is None:
            raise ValueError("Dynamic times not specified.")
        elif times is not None and times_from_file is None:
            self._times = np.array(eval_range(times), dtype= float)
        elif times is None and times_from_file is not None:
            self._times = load_table(times_from_file).to_numpy(dtype=float)
        elif times is not None and times_from_file is not None:
            raise ValueError("Both times and times_from_file specified. Choose one.")
        else:
            raise NotImplementedError
        self._non_destructive_readout = non_destructive_readout

        if np.ndim(times) != 1:
            raise ValueError("Number of dimensions in the times array is not 1.")

        self._steps = np.concatenate((times[:1], np.diff(times)), axis=0)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__  # type: str
        return f"{cls_name}<outputs={self.outputs!r}>"

    @property
    def times(self):
        """TBW."""
        return self._times

    @property
    def steps(self):
        """TBW."""
        return self._steps

    def time_it(self):
        """

        Returns
        -------

        """
        for time, step in zip(self.times, self.steps):
            yield time, step

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
            steps=self._steps,
            time_step=self._t_step,
            ndreadout=self._non_destructive_readout,
        )
        # The detector should be reset before exposure
        detector.initialize(reset_all=True)

        # prepare lists for to-be-merged datasets
        list_datasets = []

        # Dimensions set by the detectors dimensions
        rows, columns = (
            processor.detector.geometry.row,
            processor.detector.geometry.row,
        )
        # Coordinates
        coordinates = {"x": range(columns), "y": range(rows)}

        pbar = tqdm(total=self._steps)
        # TODO: Use an iterator for that ?
        while detector.elapse_time():
            logging.info("time = %.3f s", detector.time)
            if detector.is_non_destructive_readout:
                detector.initialize(reset_all=False)
            else:
                detector.initialize(reset_all=True)
            processor.run_pipeline()
            if detector.read_out:
                self.outputs.save_to_file(processor)
            # Saving all arrays into an xarray dataset for possible
            # display with holoviews in jupyter notebook
            # Initialize an xarray dataset
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

        # result = DynamicResult(
        #     dataset=final_dataset,
        #     # parameters=final_parameters_merged,
        #     # logs=final_logs,
        # )

        return final_dataset

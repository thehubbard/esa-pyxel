#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
import operator
import typing as t

import xarray as xr

from .dynamic import Dynamic, dynamic_pipeline

if t.TYPE_CHECKING:
    from ..inputs_outputs import ObservationOutputs
    from ..pipelines import Processor


class Observation:
    """TBW."""

    def __init__(
        self, outputs: "ObservationOutputs", dynamic: t.Optional[Dynamic] = None
    ):
        self.outputs = outputs
        self.dynamic = dynamic

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__  # type: str
        return f"{cls_name}<outputs={self.outputs!r}>"

    def run_observation(self, processor: "Processor") -> xr.Dataset:
        result = run_observation(
            processor=processor,
            dynamic=self.dynamic,
            outputs=self.outputs,
            dynamic_progressbar=True,
        )
        return result


def run_observation(
    processor: "Processor",
    dynamic: t.Optional["Dynamic"] = None,
    outputs: t.Optional["ObservationOutputs"] = None,
    dynamic_progressbar: bool = False,
) -> xr.Dataset:
    """Run a single or dynamic pipeline.

    Parameters
    ----------
    processor
    dynamic
    outputs
    dynamic_progressbar

    Returns
    -------
    result: xr.Dataset
    """
    if not dynamic:
        result = single_pipeline(processor=processor, outputs=outputs)

    else:
        result = dynamic_pipeline(
            processor=processor,
            time_step_it=dynamic.time_step_it(),
            num_steps=dynamic._num_steps,
            ndreadout=dynamic.non_destructive_readout,
            times_linear=dynamic._times_linear,
            start_time=dynamic._start_time,
            end_time=dynamic._times[-1],
            outputs=outputs,
            progressbar=dynamic_progressbar,
        )

    return result


def single_pipeline(
    processor: "Processor", outputs: t.Optional["ObservationOutputs"] = None
) -> xr.Dataset:
    """Run a single pipeline and return the result dataset.

    Parameters
    ----------
    processor: Processor
    outputs: ObservationOutputs

    Returns
    -------
    dataset: xr.Dataset
    """
    _ = processor.run_pipeline()

    if outputs:
        outputs.save_to_file(processor)

    detector = processor.detector

    rows, columns = (
        detector.geometry.row,
        detector.geometry.row,
    )

    coordinates = {"x": range(columns), "y": range(rows)}

    dataset = xr.Dataset()

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

        dataset[key] = da

    return dataset

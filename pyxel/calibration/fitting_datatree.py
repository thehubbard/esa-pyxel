#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

""":term:`CDM` model calibration with PYGMO.

https://esa.github.io/pagmo2/index.html
"""
import copy
import logging
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from dask.delayed import delayed
from numpy.typing import NDArray

from pyxel.calibration import (
    CalibrationMode,
    FittingCallable,
    ProblemSingleObjective,
    check_ranges,
    read_data,
    read_datacubes,
)
from pyxel.exposure import run_pipeline
from pyxel.observation import ParameterValues
from pyxel.pipelines import Processor, ResultType

if TYPE_CHECKING:
    import xarray as xr
    from datatree import DataTree
    from numpy.typing import ArrayLike

    from pyxel.exposure import Readout


@dataclass
class FitRange2D:
    """Represent a 2D range or slice with a row range and a column range.

    Parameters
    ----------
    row : slice
        Range of rows in the 2D range
    col : slice
        Range of columns in the 2D range

    Examples
    --------
    >>> range2d = FitRange2D(row=slice(0, 5), col=slice(2, 7))
    >>> row_slice, col_slice = range2d.to_slices()
    >>> row_slice
    slice(0, 5)
    >>> col_slice
    slice(2,7)

    >>> Range2D.from_sequence([0, 5, 2, 7])
    FitRange2D(slice(0, 5),slice(2,7))
    """

    row: slice
    col: slice

    @classmethod
    def from_sequence(cls, data: Sequence[Optional[int]]) -> "FitRange2D":
        if not data:
            data = [None] * 4

        if len(data) != 4:
            raise ValueError("Fitting range should have 4 values")

        y_start, y_stop, x_start, x_stop = data
        return cls(row=slice(y_start, y_stop), col=slice(x_start, x_stop))

    def to_dict(self) -> Mapping[str, slice]:
        return {"y": self.row, "x": self.col}

    def to_slices(self) -> tuple[slice, slice]:
        return self.row, self.col


@dataclass
class FitRange3D:
    """Represent a 3D range or slice with a time range, row range and a column range.

    Parameters
    ----------
    time : FitSlice
         Range of time in the 3D range
    row : FitSlice
        Range of rows in the 3D range
    col : FitSlice
        Range of columns in the 3D range

    Examples
    --------
    >>> range3d = FitRange3D(time=slice(0, 10), row=slice(0, 5), col=slice(2, 7))
    >>> time_slice, row_slice, col_slice = range3d.to_slices()
    >>> time_slice
    slice(0, 10)
    >>> row_slice
    slice(0, 5)
    >>> col_slice
    slice(2,7)

    >>> FitRange3D.from_sequence([0, 10, 0, 5, 2, 7])
    FitRange3D(time=slice(0,10), row=slice(0, 5), col=slice(2,7))
    """

    time: slice
    row: slice
    col: slice

    @classmethod
    def from_sequence(cls, data: Sequence[Optional[int]]) -> "FitRange3D":
        if not data:
            data = [None] * 6

        if len(data) == 4:
            data = [None, None, *data]

        if len(data) != 6:
            raise ValueError("Fitting range should have 6 values")

        time_start, time_stop, y_start, y_stop, x_start, x_stop = data
        return cls(
            time=slice(time_start, time_stop),
            row=slice(y_start, y_stop),
            col=slice(x_start, x_stop),
        )

    def to_dict(self) -> Mapping[str, slice]:
        return {"time": self.time, "y": self.row, "x": self.col}

    def to_slices(self) -> tuple[slice, slice, slice]:
        return self.time, self.row, self.col


def list_to_fit_range(
    input_list: Optional[Sequence[int]] = None,
) -> Union[FitRange2D, FitRange3D]:
    if not input_list:
        return FitRange2D(row=slice(None), col=slice(None))

    elif len(input_list) == 4:
        return FitRange2D.from_sequence(input_list)

    elif len(input_list) == 6:
        return FitRange3D.from_sequence(input_list)

    else:
        raise ValueError("Fitting range should have 4 or 6 values")


class ModelFittingDataTree(ProblemSingleObjective):
    """Pygmo problem class to fit data with any model in Pyxel."""

    def __init__(
        self,
        processor: Processor,
        variables: Sequence[ParameterValues],
        readout: "Readout",
        calibration_mode: CalibrationMode,
        simulation_output: ResultType,
        generations: int,
        population_size: int,
        fitness_func: FittingCallable,
        file_path: Path,
        target_fit_range: Sequence[int],
        out_fit_range: Sequence[int],
        target_output: Sequence[Path],
        input_arguments: Optional[Sequence[ParameterValues]] = None,
        weights: Optional[Sequence[float]] = None,
        weights_from_file: Optional[Sequence[Path]] = None,
        pipeline_seed: Optional[int] = None,
    ):
        self.processor: Processor = processor
        self.variables: Sequence[ParameterValues] = variables

        self.calibration_mode: CalibrationMode = calibration_mode
        # self.original_processor: Optional[Processor] = None
        self.generations: int = generations
        self.pop: int = population_size
        self.readout: Readout = readout

        self.all_target_data: list[np.ndarray] = []
        self.weighting: Optional[np.ndarray] = None
        self.weighting_from_file: Optional[Sequence[np.ndarray]] = None
        self.fitness_func: FittingCallable = fitness_func
        self.sim_output: ResultType = simulation_output
        self.param_processor_list: list[Processor] = []

        self.file_path: Path = file_path
        self.pipeline_seed: Optional[int] = pipeline_seed

        self.fitness_array: Optional[np.ndarray] = None
        self.population: Optional[np.ndarray] = None
        # self.champion_f_list: Optional[np.ndarray] = None
        # self.champion_x_list: Optional[np.ndarray] = None

        self.lbd: list[float] = []  # lower boundary
        self.ubd: list[float] = []  # upper boundary

        # self.sim_fit_range: Union[FitRange2D, FitRange3D, None] = None
        # self.targ_fit_range: Union[FitRange2D, FitRange3D, None] = None

        self.match: dict[int, list[str]] = {}

        # if self.calibration_mode == 'single_model':           # TODO update
        #     self.single_model_calibration()

        self.set_bound()

        self.original_processor: Processor = deepcopy(self.processor)

        if input_arguments:
            max_val, min_val = 0, 1000
            for arg in input_arguments:
                min_val = min(min_val, len(arg.values))
                max_val = max(max_val, len(arg.values))
            if min_val != max_val:
                logging.warning(
                    'The "result_input_arguments" value lists have different lengths! '
                    "Some values will be ignored."
                )
            for i in range(min_val):
                new_processor: Processor = deepcopy(self.processor)

                step: ParameterValues
                for step in input_arguments:
                    assert step.values != "_"

                    value: Union[Literal["_"], str, Number] = step.values[i]

                    step.current = value
                    new_processor.set(key=step.key, value=step.current)
                self.param_processor_list += [new_processor]
        else:
            self.param_processor_list = [deepcopy(self.processor)]

        params: int = 0

        var: ParameterValues
        for var in self.variables:
            if isinstance(var.values, list):
                b = len(var.values)
            else:
                b = 1

            params += b
        self.champion_f_list: np.ndarray = np.zeros((1, 1))
        self.champion_x_list: np.ndarray = np.zeros((1, params))

        if self.readout.time_domain_simulation:
            target_list_3d: Sequence[np.ndarray] = read_datacubes(
                filenames=target_output
            )
            times, rows, cols = target_list_3d[0].shape

            # TODO: Create a new function 'check_fit_ranges'
            check_ranges(
                target_fit_range=target_fit_range,
                out_fit_range=out_fit_range,
                rows=rows,
                cols=cols,
                readout_times=times,
            )

            self.targ_fit_range: Union[FitRange2D, FitRange3D] = list_to_fit_range(
                target_fit_range
            )
            self.sim_fit_range: FitRange3D = FitRange3D.from_sequence(out_fit_range)

            target_3d: np.ndarray
            for target_3d in target_list_3d:
                self.all_target_data += [target_3d[self.targ_fit_range.to_slices()]]

        else:
            target_list_2d: Sequence[np.ndarray] = read_data(filenames=target_output)

            rows, cols = target_list_2d[0].shape
            check_ranges(
                target_fit_range=target_fit_range,
                out_fit_range=out_fit_range,
                rows=rows,
                cols=cols,
            )
            self.targ_fit_range = list_to_fit_range(target_fit_range)
            self.sim_fit_range = FitRange3D.from_sequence(out_fit_range)

            target_2d: np.ndarray
            for target_2d in target_list_2d:
                self.all_target_data += [target_2d[self.targ_fit_range.to_slices()]]

            self._configure_weights(
                weights=weights,
                weights_from_file=weights_from_file,
            )

    def get_bounds(self) -> tuple[Sequence[float], Sequence[float]]:
        """Get the box bounds of the problem (lower_boundary, upper_boundary).

        It also implicitly defines the dimension of the problem.

        Returns
        -------
        tuple of lower boundaries and upper boundaries
        """
        return self.lbd, self.ubd

    def _configure_weights(
        self,
        weights: Optional[Sequence[float]] = None,
        weights_from_file: Optional[Sequence[Path]] = None,
    ) -> None:
        """TBW.

        Parameters
        ----------
        weights
        weights_from_file
        """
        if weights_from_file is not None:
            if self.readout.time_domain_simulation:
                wf = read_datacubes(weights_from_file)
                self.weighting_from_file = [
                    weight_array[self.targ_fit_range.to_slices()] for weight_array in wf
                ]
            else:
                wf = read_data(weights_from_file)
                self.weighting_from_file = [
                    weight_array[self.targ_fit_range.to_slices()] for weight_array in wf
                ]
        elif weights is not None:
            self.weighting = np.array(weights)

    def set_bound(self) -> None:
        """TBW."""
        self.lbd = []
        self.ubd = []

        var: ParameterValues
        for var in self.variables:
            assert var.boundaries is not None  # TODO: Fix this

            if var.values == "_":
                assert var.boundaries.shape == (2,)  # TODO: Fix this

                low_val: float
                high_val: float
                low_val, high_val = var.boundaries

                if var.logarithmic:
                    low_val = math.log10(low_val)
                    high_val = math.log10(high_val)

                self.lbd += [low_val]
                self.ubd += [high_val]

            elif isinstance(var.values, Sequence) and all(
                x == "_" for x in var.values[:]
            ):
                if var.boundaries.ndim == 1:
                    low_val, high_val = var.boundaries

                    low_values: NDArray[np.float_] = np.array(
                        [low_val] * len(var.values)
                    )
                    high_values: NDArray[np.float_] = np.array(
                        [high_val] * len(var.values)
                    )

                elif var.boundaries.ndim == 2:
                    low_values = var.boundaries[:, 0]
                    high_values = var.boundaries[:, 1]
                else:
                    raise NotImplementedError

                if var.logarithmic:
                    low_values = np.log10(low_values)
                    high_values = np.log10(high_values)

                self.lbd += low_values.tolist()
                self.ubd += high_values.tolist()

            else:
                raise ValueError(
                    'Character "_" (or a list of it) should be used to '
                    "indicate variables need to be calibrated"
                )

    def get_simulated_data(self, data: "DataTree") -> "xr.DataArray":
        """Extract 2D data from a processor."""
        import xarray as xr

        if self.sim_output == ResultType.Image:
            simulated_data = data["bucket/image"]

        elif self.sim_output == ResultType.Signal:
            simulated_data = data["bucket/signal"]

        elif self.sim_output == ResultType.Pixel:
            simulated_data = data["bucket/pixel"]
        else:
            raise NotImplementedError(
                f"Simulation mode: {self.sim_output!r} not implemented"
            )

        if not isinstance(simulated_data, xr.DataArray):
            raise ValueError("Expected a 'DataArray'")

        simulated_data = simulated_data.sel(self.sim_fit_range.to_dict())

        return simulated_data

    def calculate_fitness(
        self,
        simulated_data: npt.ArrayLike,
        target_data: np.ndarray,
        weighting: Optional[np.ndarray] = None,
    ) -> float:
        if weighting is not None:
            factor = weighting
        else:
            factor = np.ones(np.shape(target_data))

        fitness: float = self.fitness_func(
            simulated=np.array(simulated_data, dtype=float),
            target=np.array(target_data, dtype=float),
            weighting=np.array(factor, dtype=float),
        )

        return fitness

    # TODO: If possible, use 'numba' for this method
    def fitness(self, decision_vector_1d: np.ndarray) -> Sequence[float]:
        """Call the fitness function, elements of parameter array could be logarithmic values.

        Parameters
        ----------
        decision_vector_1d : array_like
            A 1d decision vector.

        Returns
        -------
        sequence
            The fitness of the input decision vector (concatenating the objectives,
            the equality and the inequality constraints)
        """
        # TODO: Fix this
        if self.pop is None:
            raise NotImplementedError("'pop' is not initialized.")

        try:
            # TODO: Use directory 'logging.'
            logger = logging.getLogger("pyxel")
            prev_log_level = logger.getEffectiveLevel()

            parameter_1d = self.convert_to_parameters(decision_vector_1d)
            # TODO: deepcopy is not needed. Check this
            processor_list: Sequence[Processor] = self.param_processor_list

            overall_fitness: float = 0.0
            for i, (processor, target_data) in enumerate(
                zip(processor_list, self.all_target_data)
            ):
                processor = self.update_processor(
                    parameter=parameter_1d, processor=processor
                )

                logger.setLevel(logging.WARNING)
                # result_proc = None
                if self.calibration_mode != CalibrationMode.Pipeline:
                    raise NotImplementedError

                data_tree: "DataTree" = run_pipeline(
                    processor=processor,
                    readout=self.readout,
                    pipeline_seed=self.pipeline_seed,
                )
                # elif self.calibration_mode == 'single_model':
                #     self.fitted_model.function(processor.detector)               # todo: update

                logger.setLevel(prev_log_level)

                simulated_data: "xr.DataArray" = self.get_simulated_data(data=data_tree)

                weighting: Optional[np.ndarray] = None

                if self.weighting is not None:
                    weighting = np.full(
                        shape=(
                            processor.detector.geometry.row,
                            processor.detector.geometry.col,
                        ),
                        fill_value=self.weighting[i],
                    )
                elif self.weighting_from_file is not None:
                    weighting = self.weighting_from_file[i]

                overall_fitness += self.calculate_fitness(
                    simulated_data=simulated_data,
                    target_data=target_data,
                    weighting=weighting,
                )

        except Exception:
            logging.exception(
                "Catch an exception in 'fitness' for ModelFitting: %r.", self
            )
            raise

        return [overall_fitness]

    def convert_to_parameters(self, decisions_vector: "ArrayLike") -> np.ndarray:
        """Convert a decision version from Pygmo2 to parameters.

        Parameters
        ----------
        decisions_vector : array_like
            It could a 1D or 2D array.

        Returns
        -------
        array_like
            Parameters
        """
        parameters = np.array(decisions_vector)

        a = 0
        for var in self.variables:
            b = 1
            if isinstance(var.values, list):
                b = len(var.values)
            if var.logarithmic:
                start = a
                stop = a + b
                parameters[..., start:stop] = np.power(10, parameters[..., start:stop])
            a += b

        return parameters

    def apply_parameters(
        self, processor: Processor, parameter: np.ndarray
    ) -> "DataTree":
        """Create a new ``Processor`` with new parameters."""
        new_processor = self.update_processor(parameter=parameter, processor=processor)

        data_tree: "DataTree" = run_pipeline(
            processor=new_processor,
            readout=self.readout,
            pipeline_seed=self.pipeline_seed,
        )

        return data_tree

    def apply_parameters_to_processors(
        self, parameters: "xr.DataArray"
    ) -> pd.DataFrame:
        """TBW."""
        assert "island" in parameters.dims
        assert "param_id" in parameters.dims

        lst = []
        for id_processor, processor in enumerate(self.param_processor_list):
            delayed_processor = delayed(processor)

            for idx_island, params_array in parameters.groupby("island"):
                params: np.ndarray = params_array.data  # type

                result_datatree: DataTree = delayed(self.apply_parameters)(
                    processor=delayed_processor, parameter=params
                )

                lst.append(
                    {
                        "island": idx_island,
                        "id_processor": id_processor,
                        "data_tree": result_datatree,
                    }
                )

        df = pd.DataFrame(lst).sort_values(["island", "id_processor"])

        return df

    def update_processor(
        self, parameter: np.ndarray, processor: Processor
    ) -> Processor:
        """TBW."""
        new_processor = copy.deepcopy(processor)
        a, b = 0, 0
        for var in self.variables:
            if var.values == "_":
                b = 1
                new_processor.set(key=var.key, value=parameter[a])
            elif isinstance(var.values, list):
                b = len(var.values)

                start = a
                stop = a + b
                new_processor.set(key=var.key, value=parameter[start:stop])
            a += b
        return new_processor

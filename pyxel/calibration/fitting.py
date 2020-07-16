#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""CDM model calibration with PYGMO.

https://esa.github.io/pagmo2/index.html
"""
import logging
import math
import os
import re
import typing as t
from collections import OrderedDict
from copy import deepcopy
from numbers import Number
from operator import add
from pathlib import Path

import numpy as np
from dask import delayed
from dask.delayed import Delayed

from pyxel.calibration.util import (
    CalibrationMode,
    ResultType,
    check_ranges,
    list_to_slice,
    read_data,
)
from pyxel.parametric.parameter_values import ParameterValues
from pyxel.pipelines import Processor


class ModelFitting:
    """Pygmo problem class to fit data with any model in Pyxel."""

    def __init__(self, processor: Processor, variables: t.Sequence[ParameterValues]):
        """TBW."""
        self.processor = processor  # type: Processor
        self.variables = variables  # type: t.Sequence[ParameterValues]

        self.calibration_mode = None  # type: t.Optional[CalibrationMode]
        self.original_processor = None  # type: t.Optional[Processor]
        self.generations = None  # type: t.Optional[int]
        self.pop = None  # type: t.Optional[int]

        self.all_target_data = []  # type: t.List[np.ndarray]
        self.weighting = None  # type: t.Optional[np.ndarray]
        self.fitness_func = None  # type: t.Optional[t.Callable]
        self.sim_output = None  # type: t.Optional[ResultType]
        # self.fitted_model = None            # type: t.Optional['ModelFunction']
        self.param_processor_list = []  # type: t.List[Processor]

        self.n = 0  # type: int
        self.g = 0  # type: int

        self.file_path = None  # type: t.Optional[Path]

        self.fitness_array = None  # type: t.Optional[np.ndarray]
        self.population = None  # type: t.Optional[np.ndarray]
        self.champion_f_list = None  # type: t.Optional[np.ndarray]
        self.champion_x_list = None  # type: t.Optional[np.ndarray]

        self.lbd = []  # type: t.List[float]  # lower boundary
        self.ubd = []  # type: t.List[float]  # upper boundary

        self.sim_fit_range = slice(None)  # type: t.Union[slice, t.Tuple[slice, slice]]
        self.targ_fit_range = slice(None)  # type: t.Union[slice, t.Tuple[slice, slice]]

        self.match = {}  # type: t.Dict[int, t.List[str]]

        # self.normalization = False
        # self.target_data_norm = []

    def get_bounds(self) -> t.Tuple[t.Sequence[float], t.Sequence[float]]:
        """TBW.

        :return:
        """
        return self.lbd, self.ubd

    def configure(
        self,
        calibration_mode: CalibrationMode,
        simulation_output: ResultType,
        fitness_func: t.Callable,
        population_size: int,
        generations: int,
        file_path: t.Optional[Path],
        target_fit_range: t.Sequence[int],
        out_fit_range: t.Sequence[int],
        target_output: t.Sequence[Path],
        input_arguments: t.Optional[t.Sequence[ParameterValues]] = None,
        weighting: t.Optional[t.Sequence[Path]] = None,
    ) -> None:
        """TBW.

        Parameters
        ----------
        calibration_mode
        simulation_output
        fitness_func
        population_size
        generations
        file_path
        target_fit_range
        out_fit_range
        target_output
        input_arguments
        weighting
        """
        self.calibration_mode = CalibrationMode(calibration_mode)
        self.sim_output = ResultType(simulation_output)
        self.fitness_func = fitness_func
        self.pop = population_size
        self.generations = generations

        # TODO: Remove 'assert'
        # assert isinstance(self.pop, int)

        # if self.calibration_mode == 'single_model':           # TODO update
        #     self.single_model_calibration()

        self.set_bound()

        self.original_processor = deepcopy(self.processor)
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
                new_processor = deepcopy(self.processor)  # type: Processor
                for step in input_arguments:  # type: ParameterValues
                    assert step.values != "_"

                    value = step.values[i]  # type: t.Union[t.Literal['_'], str, Number]

                    step.current = value
                    new_processor.set(key=step.key, value=step.current)
                self.param_processor_list += [new_processor]
        else:
            self.param_processor_list = [deepcopy(self.processor)]

        params = 0  # type: int
        for var in self.variables:  # type: ParameterValues
            if isinstance(var.values, list):
                b = len(var.values)
            else:
                b = 1

            params += b
        self.champion_f_list = np.zeros((1, 1))
        self.champion_x_list = np.zeros((1, params))
        self.file_path = file_path

        target_list = read_data(filenames=target_output)  # type: t.Sequence[np.ndarray]
        try:
            rows, cols = target_list[0].shape  # type: t.Tuple[int, t.Optional[int]]
        except AttributeError:
            rows = len(target_list[0])
            cols = None
        check_ranges(
            target_fit_range=target_fit_range,
            out_fit_range=out_fit_range,
            rows=rows,
            cols=cols,
        )
        self.targ_fit_range = list_to_slice(target_fit_range)
        self.sim_fit_range = list_to_slice(out_fit_range)
        for target in target_list:  # type: np.ndarray
            self.all_target_data += [target[self.targ_fit_range]]

        if weighting:
            wf = read_data(weighting)[0]  # type: np.ndarray
            self.weighting = wf[self.targ_fit_range]

    # def single_model_calibration(self):     # TODO update
    #     """TBW.
    #
    #     :return:
    #     """
    #     # if len(self.model_name_list) > 1:
    #     #     raise ValueError('Select only one pipeline model!')
    #     # if self.model_name_list[0] in ['geometry', 'material', 'environment', 'characteristics']:
    #     #     raise ValueError('Select a pipeline model and not a detector attribute!')
    #
    #     self.fitted_model = self.processor.pipeline.get_model(self.model_name_list[0])
    #     self.processor.run_pipeline(abort_before=self.model_name_list[0])

    def set_bound(self) -> None:
        """TBW."""
        self.lbd = []
        self.ubd = []
        for var in self.variables:  # type: ParameterValues
            assert var.boundaries
            low_val, high_val = var.boundaries  # type: t.Tuple[float, float]

            if var.logarithmic:
                low_val = math.log10(low_val)
                high_val = math.log10(high_val)

            if var.values == "_":
                self.lbd += [low_val]
                self.ubd += [high_val]
            elif isinstance(var.values, list) and all(x == "_" for x in var.values[:]):
                self.lbd += [low_val] * len(var.values)
                self.ubd += [high_val] * len(var.values)
            else:
                raise ValueError(
                    'Character "_" (or a list of it) should be used to '
                    "indicate variables need to be calibrated"
                )

    def get_simulated_data(self, processor: Processor) -> np.ndarray:
        """TBW."""
        if self.sim_output == ResultType.Image:
            simulated_data = processor.detector.image.array[
                self.sim_fit_range
            ]  # type: np.ndarray

        elif self.sim_output == ResultType.Signal:
            simulated_data = processor.detector.signal.array[self.sim_fit_range]
        elif self.sim_output == ResultType.Pixel:
            simulated_data = processor.detector.pixel.array[self.sim_fit_range]

        return simulated_data

    def batch_fitness(self, population_parameter_vector):
        """Batch Fitness Evaluation.

        PYGMO BFE IS STILL NOT FULLY IMPLEMENTED, THEREFORE THIS FUNC CAN NOT BE USED YET.
        """
        logger = logging.getLogger("pyxel")
        logger.info("batch_fitness() called with %s " % population_parameter_vector)

        fitness_vector = []
        for parameter in population_parameter_vector:
            overall_fitness = 0.0
            parameter = self.update_parameter([parameter])
            processor_list = deepcopy(self.param_processor_list)
            for processor, target_data in zip(processor_list, self.all_target_data):
                # processor = self.update_processor(parameter, processor)
                processor = delayed(self.update_processor)(parameter, processor)
                # result_proc = processor.run_pipeline()
                result_proc = delayed(processor.run_pipeline)()
                # simulated_data = self.get_simulated_data(result_proc)
                simulated_data = delayed(self.get_simulated_data)(result_proc)
                # fitness = self.calculate_fitness(simulated_data, target_data)
                fitness = delayed(self.calculate_fitness)(simulated_data, target_data)
                # overall_fitness = add(overall_fitness, fitness)
                overall_fitness = delayed(add)(overall_fitness, fitness)

            fitness_vector.append(
                overall_fitness
            )  # overall fitness per individual for the full population

        # fitness_vector = self.merge(fitness_vector)
        fitness_vector_delayed = delayed(merge_fitness)(fitness_vector)  # type: Delayed
        # population_fitness_vector = fitness_vector
        population_fitness_vector = fitness_vector_delayed.compute()

        return population_fitness_vector

    def calculate_fitness(
        self, simulated_data: np.ndarray, target_data: np.ndarray
    ) -> float:
        """TBW.

        :param simulated_data:
        :param target_data:
        :return:
        """
        # TODO: Remove 'assert'
        assert self.fitness_func is not None

        fitness = self.fitness_func(
            simulated=simulated_data, target=target_data, weighting=self.weighting
        )  # type: float

        return fitness

    def fitness(self, parameter: np.ndarray) -> t.Sequence[float]:
        """Call the fitness function, elements of parameter array could be logarithmic values.

        :param parameter: 1d np.array
        :return:
        """
        # TODO: Fix this
        if self.pop is None:
            raise NotImplementedError("'pop' is not initialized.")

        # TODO: Use directory 'logging.'
        logger = logging.getLogger("pyxel")
        prev_log_level = logger.getEffectiveLevel()

        parameter = self.update_parameter(parameter)
        processor_list = deepcopy(self.param_processor_list)  # type: t.List[Processor]

        overall_fitness = 0.0  # type: float
        for processor, target_data in zip(processor_list, self.all_target_data):

            processor = self.update_processor(
                parameter=parameter, new_processor=processor
            )

            logger.setLevel(logging.WARNING)
            # result_proc = None
            if self.calibration_mode == CalibrationMode.Pipeline:
                result_proc = processor.run_pipeline()  # type: Processor
            # elif self.calibration_mode == 'single_model':
            #     self.fitted_model.function(processor.detector)               # todo: update
            else:
                raise NotImplementedError

            logger.setLevel(prev_log_level)

            simulated_data = self.get_simulated_data(processor=result_proc)

            overall_fitness += self.calculate_fitness(
                simulated_data=simulated_data, target_data=target_data
            )

        self.save_population(parameter=parameter, overall_fitness=overall_fitness)

        if (self.n + 1) % self.pop == 0:
            logger.info("%d. generation", self.g)
            self.champion_to_file(parameter)
            self.g += 1

        self.n += 1

        return [overall_fitness]

    # TODO: This method changes the input 'parameter'. Is it normal ?
    def update_parameter(self, parameter: np.ndarray) -> np.ndarray:
        """TBW.

        :param parameter: 1d np.array
        :return:
        """
        a = 0
        for var in self.variables:
            b = 1
            if isinstance(var.values, list):
                b = len(var.values)
            if var.logarithmic:
                start = a
                stop = a + b
                parameter[start:stop] = np.power(10, parameter[start:stop])
            a += b
        return parameter

    def update_processor(
        self, parameter: np.ndarray, new_processor: Processor
    ) -> Processor:
        """TBW.

        :param parameter:
        :param new_processor:
        :return:
        """
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

    def file_matching_renaming(self) -> None:
        """TBW."""
        if self.file_path:
            # TODO: Sort these filenames !
            output_champion_files = self.file_path.glob(
                "champions_id_*.out"
            )  # type: t.Iterator[Path]

            for ii, chfile in enumerate(output_champion_files):
                # Get last line from file 'chfile'
                with chfile.open() as fh:
                    *_, lastline = fh.readlines()

                cleaned_lastline = lastline.replace("\n", "")  # type: str
                columns = cleaned_lastline.split(" ")  # type: t.List[str]

                _, *other_columns = columns
                self.match[ii] = other_columns

                os.rename(chfile, self.file_path.joinpath(f"champions_id{ii}.out"))

                result = re.findall(
                    pattern=r"champions_id_(\w+)\.out", string=chfile.name
                )
                if len(result) != 1:
                    raise ValueError(f"Cannot extract information from '{result}'.")
                aw = result[0]  # type: str

                fid = aw.split("_")[-1]  # type: str
                popfiles = list(
                    self.file_path.glob(f"population_id_{fid}.out")
                )  # type: t.List[Path]

                popfile = popfiles[0]  # type: Path

                os.rename(popfile, self.file_path.joinpath(f"population_id{ii}.out"))

    def get_results(
        self, overall_fitness: np.ndarray, parameter: np.ndarray
    ) -> t.Tuple[t.List[Processor], t.Dict[str, t.Union[int, float]]]:
        """TBW.

        :param overall_fitness:
        :param parameter:
        :return:
        """
        results = OrderedDict()  # type: t.Dict[str, t.Union[int, float]]
        results["fitness"] = overall_fitness[0]

        # TODO: Apply a copy of 'parameter' in 'self.update_parameter' ??
        parameter = self.update_parameter(parameter)

        if self.file_path:
            # TODO: Use 'np.ravel(parameter)' instead of 'parameter.reshape(1, len(parameter))' ?
            arr_2d = np.c_[
                overall_fitness, parameter.reshape(1, len(parameter))
            ]  # type: np.ndarray

            with np.printoptions(formatter={"float": "{: .6E}".format}, suppress=False):
                arr_str = np.array2string(
                    arr_2d, separator="", suppress_small=False
                )  # type: str

            arr_str = (
                arr_str.replace("\n", "")
                .replace("   ", " ")
                .replace("  ", " ")
                .replace("[[ ", "")
                .replace("]]", "")
            )

            lst_str = arr_str.split(" ")  # type: t.Sequence[str]
            island = -1  # type: int
            for k, v in self.match.items():
                if lst_str == v:
                    island = k
                    break
            if island == -1:
                raise RuntimeError()
        else:
            island = 0
        results["island"] = island
        logging.info(
            "Post-processing island %d, champion fitness: %1.5e",
            island,
            overall_fitness[0],
        )

        champion_list = deepcopy(self.param_processor_list)  # type: t.List[Processor]
        for processor in champion_list:
            processor = self.update_processor(
                parameter=parameter, new_processor=processor
            )
            if self.calibration_mode is CalibrationMode.Pipeline:
                processor.run_pipeline()

        a, b = 0, 0
        for var in self.variables:
            if var.values == "_":
                b = 1
                results[var.key] = parameter[a]
            elif isinstance(var.values, list):
                b = len(var.values)

                start = a
                stop = a + b
                results[var.key] = parameter[start:stop]
            a += b

        return champion_list, results

    def champion_to_file(self, parameter: np.ndarray) -> None:
        """Get champion of each generation and write it to output files together with last population.

        :return:
        """
        if self.champion_f_list is None:
            raise RuntimeError(
                "'champion_f_list' was not initialized with method '.configure(...)'."
            )

        if self.champion_x_list is None:
            raise RuntimeError(
                "'champion_x_list' was not initialized with method '.configure(...)'."
            )

        if self.fitness_array is None:
            raise RuntimeError(
                "'fitness_array' was not initialized with method '.configure(...)'."
            )

        if self.population is None:
            raise RuntimeError(
                "'population' was not initialized with method '.configure(...)'."
            )

        best_index = np.argmin(self.fitness_array)  # type: int

        if self.g == 0:
            self.champion_f_list[self.g] = self.fitness_array[best_index]
            self.champion_x_list[self.g] = self.population[best_index, :]
        else:
            best_champ_index = np.argmin(self.champion_f_list)

            if self.fitness_array[best_index] < self.champion_f_list[best_champ_index]:
                self.champion_f_list = np.vstack(
                    (self.champion_f_list, self.fitness_array[best_index])
                )
                self.champion_x_list = np.vstack(
                    (self.champion_x_list, self.population[best_index])
                )
            else:
                self.champion_f_list = np.vstack(
                    (self.champion_f_list, self.champion_f_list[-1])
                )
                self.champion_x_list = np.vstack(
                    (self.champion_x_list, self.champion_x_list[-1])
                )

        # TODO: should we keep and write to file the population(s) which had the champion inside?
        # because usually this is not the last population currently we save to file!

        if self.file_path and self.g > 0:
            self.add_to_champ_file(parameter)
            self.add_to_pop_file(parameter)

    def save_population(self, parameter: np.ndarray, overall_fitness: float) -> None:
        """Save population of each generation to get champions.

        :param parameter: 1d np.array
        :param overall_fitness: list
        :return:
        """
        if self.pop is None:
            raise RuntimeError("'pop' was not initialized with method '.configure'.")

        if self.n % self.pop == 0:
            self.fitness_array = np.array([overall_fitness])
            self.population = parameter
        else:
            self.fitness_array = np.vstack(
                (self.fitness_array, np.array([overall_fitness]))
            )
            self.population = np.vstack((self.population, parameter))

    def add_to_champ_file(self, parameter: np.ndarray) -> None:
        """TBW."""
        assert self.champion_f_list is not None
        assert self.champion_x_list is not None
        assert self.file_path

        champions_file = self.file_path.joinpath(
            f"champions_id_{id(self)}.out"
        )  # type: Path
        str_format = "%d" + (len(parameter) + 1) * " %.6E"
        with champions_file.open("ab") as file1:
            np.savetxt(
                file1,
                np.c_[
                    np.array([self.g]),
                    self.champion_f_list[self.g],
                    self.champion_x_list[self.g, :].reshape(1, len(parameter)),
                ],
                fmt=str_format,
            )

    def add_to_pop_file(self, parameter: np.ndarray) -> None:
        """TBW."""
        assert self.fitness_array is not None
        assert self.file_path

        pop_file = self.file_path.joinpath(f"population_id_{id(self)}.out")
        str_format = "%d" + (len(parameter) + 1) * " %.6E"
        with pop_file.open("wb") as file2:
            np.savetxt(
                file2,
                np.c_[
                    self.g * np.ones(self.fitness_array.shape),
                    self.fitness_array,
                    self.population,
                ],
                fmt=str_format,
            )

    # def least_squares(self, simulated_data, dataset=None):
    #     """TBW.
    #
    #     :param simulated_data:
    #     :param dataset: int
    #     :return:
    #     """
    #     input_array = simulated_data[self.sim_fit_range]
    #
    #     if dataset is not None:
    #         if self.normalization:
    #             input_array = self.normalize(input_array, dataset=dataset)
    #             target = self.target_data_norm[dataset][self.targ_fit_range]
    #         else:
    #             target = self.target_data[dataset][self.targ_fit_range]
    #     else:
    #         if self.normalization:
    #             input_array = self.normalize(input_array)
    #             target = self.target_data_norm[self.targ_fit_range]
    #         else:
    #             target = self.target_data[self.targ_fit_range]
    #
    #     diff = target - input_array
    #     diff_square = diff * diff
    #     return np.sum(diff_square)

    # def set_normalization(self):
    #     """TBW.
    #
    #     :return:
    #     """
    #     self.normalization = True
    #     for i in range(len(self.target_data)):
    #         self.target_data_norm += [self.normalize(self.target_data[i], dataset=i)]

    # def normalize(self, array, dataset):
    #     """Normalize dataset arrays by injected signal maximum.
    #
    #     :param array: 1d np.array
    #     :param dataset: int
    #     :return:
    #     """
    #     return array / np.average(self.target_data[dataset][self.targ_fit_range])


def merge_fitness(f):
    """TBW."""
    return f

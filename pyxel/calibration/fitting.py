"""CDM model calibration with PYGMO.

https://esa.github.io/pagmo2/index.html
"""
import logging
from copy import deepcopy
from collections import OrderedDict
import typing as t   # noqa: F401
import numpy as np
from pyxel.calibration.util import list_to_slice, check_ranges, read_data
from pyxel.parametric.parameter_values import ParameterValues
from pyxel.pipelines.processor import Processor


# FRED: Add typing information for all methods
class ModelFitting:
    """Pygmo problem class to fit data with any model in Pyxel."""

    def __init__(self, processor, variables: t.List[ParameterValues]):
        """TBW."""
        # self.logger = logging.getLogger('pyxel')

        self.processor = processor          # type: Processor
        self.variables = variables          # type: t.List[ParameterValues]

        self.calibration_mode = None        # type: t.Optional[str]
        self.original_processor = None      # type: t.Optional[Processor]
        self.generations = None             # type: t.Optional[int]
        self.pop = None                     # type: t.Optional[int]

        self.all_target_data = []           # type: t.List[t.List[t.Any]]
        self.weighting = None               # type: t.Optional[t.List[np.ndarray]]
        self.fitness_func = None
        self.sim_output = None              # type: t.Optional[str]
        # self.fitted_model = None            # type: t.Optional['ModelFunction']
        self.param_processor_list = []      # type: t.List[Processor]

        self.n = 0
        self.g = 0

        self.champions_file = ''            # type: str
        self.pop_file = ''                  # type: str

        self.fitness_array = None
        self.population = None
        self.champion_f_list = None
        self.champion_x_list = None

        self.lbd = None                     # lower boundary
        self.ubd = None                     # upper boundary

        self.sim_fit_range = None
        self.targ_fit_range = None

        # self.normalization = False
        # self.target_data_norm = []

    def get_bounds(self):
        """TBW.

        :return:
        """
        return self.lbd, self.ubd

    # HANS: expand setting to argument names with default values.
    def configure(self, setting: dict):
        """TBW.

        :param setting: dict
        :return:
        """
        logger = logging.getLogger('pyxel')

        self.calibration_mode = setting['calibration_mode']
        self.sim_output = setting['simulation_output']
        self.fitness_func = setting['fitness_func']
        self.pop = setting['population_size']
        self.generations = setting['generations']

        # if self.calibration_mode == 'single_model':           # TODO update
        #     self.single_model_calibration()

        self.set_bound()

        self.original_processor = deepcopy(self.processor)
        if 'input_arguments' in setting and setting['input_arguments']:

            max_val, min_val = 0, 1000
            for arg in setting['input_arguments']:
                min_val = min(min_val, len(arg.values))
                max_val = max(max_val, len(arg.values))
            if min_val != max_val:
                logger.warning('The "result_input_arguments" value lists have different lengths! '
                               'Some values will be ignored.')
            for i in range(min_val):
                new_processor = deepcopy(self.processor)
                for step in setting['input_arguments']:
                    step.current = step.values[i]
                    new_processor.set(step.key, step.current)
                self.param_processor_list += [new_processor]
        else:
            self.param_processor_list = [deepcopy(self.processor)]

        params = 0
        for var in self.variables:
            b = 1
            if isinstance(var.values, list):
                b = len(var.values)
            params += b
        self.champion_f_list = np.zeros((1, 1))
        self.champion_x_list = np.zeros((1, params))
        self.champions_file = setting['champions_file']
        self.pop_file = setting['population_file']

        target_list = read_data(setting['target_output'])
        try:
            rows, cols = target_list[0].shape
        except AttributeError:
            rows = len(target_list[0])
            cols = None
        check_ranges(setting['target_fit_range'], setting['out_fit_range'], rows, cols)
        self.targ_fit_range = list_to_slice(setting['target_fit_range'])
        self.sim_fit_range = list_to_slice(setting['out_fit_range'])
        for target in target_list:
            self.all_target_data += [target[self.targ_fit_range]]

        if setting['weighting']:
            wf = read_data(setting['weighting'])[0]
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
    #     self.processor.pipeline.run_pipeline(self.processor.detector, abort_before=self.model_name_list[0])

    def set_bound(self):
        """TBW."""
        self.lbd = []
        self.ubd = []
        for var in self.variables:
            if var.logarithmic:
                var.boundaries = np.log10(var.boundaries)
            if var.values == '_':
                self.lbd += [var.boundaries[0]]
                self.ubd += [var.boundaries[1]]
            elif isinstance(var.values, list) and all(x == '_' for x in var.values[:]):
                self.lbd += [var.boundaries[0]] * len(var.values)
                self.ubd += [var.boundaries[1]] * len(var.values)
            else:
                raise ValueError('Character "_" (or a list of it) should be used to '
                                 'indicate variables need to be calibrated')

    def calculate_fitness(self, simulated_data, target_data):
        """TBW.

        :param simulated_data:
        :param target_data:
        :return:
        """
        # HANS: use this instead. The if/else statement is redundant, the weighting is optional anyways.
        fitness = self.fitness_func.function(simulated_data, target_data, self.weighting)
        # if self.weighting is not None:
        #     fitness = self.fitness_func.function(simulated_data, target_data, self.weighting)
        # else:
        #     fitness = self.fitness_func.function(simulated_data, target_data)
        return fitness

    def fitness(self, parameter):
        """Call the fitness function, elements of parameter array could be logarithmic values.

        :param parameter: 1d np.array
        :return:
        """
        parameter = self.update_parameter(parameter)
        processor_list = deepcopy(self.param_processor_list)

        overall_fitness = 0.
        for processor, target_data in zip(processor_list, self.all_target_data):

            processor = self.update_processor(parameter, processor)
            if self.calibration_mode == 'pipeline':
                processor.pipeline.run_pipeline(processor.detector)
            # elif self.calibration_mode == 'single_model':
            #     self.fitted_model.function(processor.detector)               # todo: update

            simulated_data = None
            if self.sim_output == 'image':
                simulated_data = processor.detector.image.array[self.sim_fit_range]
            elif self.sim_output == 'signal':
                simulated_data = processor.detector.signal.array[self.sim_fit_range]
            elif self.sim_output == 'pixel':
                simulated_data = processor.detector.pixel.array[self.sim_fit_range]

            overall_fitness += self.calculate_fitness(simulated_data, target_data)

        self.population_and_champions(parameter, overall_fitness)

        return [overall_fitness]

    def update_parameter(self, parameter):
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
                parameter[a:a + b] = np.power(10, parameter[a:a + b])
            a += b
        return parameter

    def update_processor(self, parameter, new_processor):
        """TBW.

        :param parameter:
        :param new_processor:
        :return:
        """
        a, b = 0, 0
        for var in self.variables:
            if var.values == '_':
                b = 1
                new_processor.set(var.key, parameter[a])
            elif isinstance(var.values, list):
                b = len(var.values)
                new_processor.set(var.key, parameter[a:a + b])
            a += b
        return new_processor

    def get_results(self, fitness, parameter):
        """TBW.

        :param fitness:
        :param parameter:
        :return:
        """
        parameter = self.update_parameter(parameter)        # todo : duplicated code, see fitness!
        new_processor = deepcopy(self.original_processor)  # TODO TODO
        champion = self.update_processor(parameter, new_processor)
        if self.calibration_mode == 'pipeline':
            champion.pipeline.run_pipeline(champion.detector)
        # elif self.calibration_mode == 'single_model':
        #     self.fitted_model.function(champion.detector)               # todo: update

        results = OrderedDict()
        results['fitness'] = fitness[0]
        a, b = 0, 0
        for var in self.variables:
            if var.values == '_':
                b = 1
                results[var.key] = parameter[a]
            elif isinstance(var.values, list):
                b = len(var.values)
                results[var.key] = parameter[a:a + b]
            a += b

        return champion, results

    def population_and_champions(self, parameter, overall_fitness):
        """Get champion (also population) of each generation and write it to output file(s).

        :param parameter: 1d np.array
        :param overall_fitness: list
        :return:
        """
        # if self.champion_file is None:
        #     self.champion_file = 'champion_id' + str(id(self)) + '.out'

        if self.n % self.pop == 0:
            self.fitness_array = np.array([overall_fitness])
            self.population = parameter
        else:
            self.fitness_array = np.vstack((self.fitness_array, np.array([overall_fitness])))
            self.population = np.vstack((self.population, parameter))

        if (self.n + 1) % self.pop == 0:

            best_index = np.argmin(self.fitness_array)

            if self.g == 0:
                self.champion_f_list[self.g] = self.fitness_array[best_index]
                self.champion_x_list[self.g] = self.population[best_index, :]
            else:
                best_champ_index = np.argmin(self.champion_f_list)

                if self.fitness_array[best_index] <= self.champion_f_list[best_champ_index]:
                    self.champion_f_list = np.vstack((self.champion_f_list, self.fitness_array[best_index]))
                    self.champion_x_list = np.vstack((self.champion_x_list, self.population[best_index]))
                else:
                    self.champion_f_list = np.vstack((self.champion_f_list, self.champion_f_list[-1]))
                    self.champion_x_list = np.vstack((self.champion_x_list, self.champion_x_list[-1]))

            str_format = '%d' + (len(parameter) + 1) * ' %.6E'

            if self.champions_file:
                with open(self.champions_file, 'ab') as file1:
                    np.savetxt(file1, np.c_[np.array([self.g]),
                                            self.champion_f_list[self.g],
                                            self.champion_x_list[self.g, :].reshape(1, len(parameter))],
                               fmt=str_format)

            if self.pop_file:
                if self.g == self.generations:
                    with open(self.pop_file, 'ab') as file2:
                        np.savetxt(file2, np.c_[self.g * np.ones(self.fitness_array.shape),
                                                self.fitness_array,
                                                self.population],
                                   fmt=str_format)

            self.g += 1

        self.n += 1

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

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""

import logging
import math
import typing as t
from enum import Enum
from pathlib import Path

import numpy as np
from pyxel.calibration.fitting import ModelFitting
from pyxel.calibration.util import CalibrationMode, ResultType
from pyxel.parametric.parameter_values import ParameterValues
from pyxel.pipelines import ModelFunction, Processor

from ..util.outputs import Outputs

try:
    import pygmo as pg
except ImportError:
    import warnings

    warnings.warn("Cannot import 'pygmo", RuntimeWarning, stacklevel=2)


class AlgorithmType(Enum):
    """TBW."""

    Sade = "sade"
    Sga = "sga"
    Nlopt = "nlopt"


# TODO: Put classes `Algorithm` and `Calibration` in separated files.
# TODO: Maybe a new class `Sade` could contains some attributes ?
# TODO: Maybe a new class `SGA` could contains some attributes ?
# TODO: Maybe a new class `NLOPT` could contains some attributes ?
class Algorithm:
    """TBW."""

    def __init__(
        self,
        # TODO: Rename 'type' into 'algorithm_type'
        type: AlgorithmType = AlgorithmType.Sade,
        generations: int = 1,
        population_size: int = 1,
        # SADE #####
        variant: int = 2,
        variant_adptv: int = 1,
        ftol: float = 1e-6,
        xtol: float = 1e-6,
        memory: bool = False,
        # SGA #####
        cr: float = 0.9,
        eta_c: float = 1.0,
        m: float = 0.02,
        param_m: float = 1.0,
        param_s: int = 2,
        crossover: str = "exponential",
        mutation: str = "polynomial",
        selection: str = "tournament",
        # NLOPT #####
        nlopt_solver: str = "neldermead",
        maxtime: int = 0,
        maxeval: int = 0,
        xtol_rel: float = 1.0e-8,
        xtol_abs: float = 0.0,
        ftol_rel: float = 0.0,
        ftol_abs: float = 0.0,
        stopval: float = -math.inf,
        local_optimizer=None,
        replacement: str = "best",
        nlopt_selection: str = "best",
    ):
        if generations not in range(1, 100001):
            raise ValueError("'generations' must be between 1 and 100000.")

        if population_size not in range(1, 100001):
            raise ValueError("'population_size' must be between 1 and 100000.")

        self._type = AlgorithmType(type)
        self._generations = generations
        self._population_size = population_size

        # SADE #####
        if variant not in range(1, 19):
            raise ValueError("'variant' must be between 1 and 18.")

        if variant_adptv not in (1, 2):
            raise ValueError("'variant_adptv' must be between 1 and 2.")

        self._variant = variant
        self._variant_adptv = variant_adptv
        self._ftol = ftol
        self._xtol = xtol
        self._memory = memory

        # SGA #####
        if not (0.0 <= cr <= 1.0):
            raise ValueError("'cr' must be between 0.0 and 1.0.")

        if not (0.0 <= m <= 1.0):
            raise ValueError("'m' must be between 0.0 and 1.0.")

        self._cr = cr
        self._eta_c = eta_c
        self._m = m
        self._param_m = param_m
        self._param_s = param_s
        self._crossover = crossover
        self._mutation = mutation
        self._selection = selection

        # NLOPT #####
        self._nlopt_solver = nlopt_solver
        self._maxtime = maxtime
        self._maxeval = maxeval
        self._xtol_rel = xtol_rel
        self._xtol_abs = xtol_abs
        self._ftol_rel = ftol_rel
        self._ftol_abs = ftol_abs
        self._stopval = stopval
        self._local_optimizer = local_optimizer
        self._replacement = replacement
        self._nlopt_selection = nlopt_selection

    @property
    def type(self) -> AlgorithmType:
        """TBW."""
        return self._type

    @type.setter
    def type(self, value: AlgorithmType) -> None:
        """TBW."""
        self._type = AlgorithmType(value)

    @property
    def generations(self) -> int:
        """TBW."""
        return self._generations

    @generations.setter
    def generations(self, value: int) -> None:
        """TBW."""
        if value not in range(1, 100001):
            raise ValueError("'generations' must be between 1 and 100000.")

        self._generations = value

    @property
    def population_size(self) -> int:
        """TBW."""
        return self._population_size

    @population_size.setter
    def population_size(self, value: int) -> None:
        """TBW."""
        if value not in range(1, 100001):
            raise ValueError("'population_size' must be between 1 and 100000.")

        self._population_size = value

    # SADE #####
    @property
    def variant(self) -> int:
        """TBW."""
        return self._variant

    @variant.setter
    def variant(self, value: int) -> None:
        """TBW."""
        if value not in range(1, 19):
            raise ValueError("'variant' must be between 1 and 18.")

        self._variant = value

    @property
    def variant_adptv(self) -> int:
        """TBW."""
        return self._variant_adptv

    @variant_adptv.setter
    def variant_adptv(self, value: int) -> None:
        """TBW."""
        if value not in (1, 2):
            raise ValueError("'variant_adptv' must be between 1 and 2.")

        self._variant_adptv = value

    @property
    def ftol(self) -> float:
        """TBW."""
        return self._ftol

    @ftol.setter
    def ftol(self, value: float) -> None:
        """TBW."""
        self._ftol = value

    @property
    def xtol(self) -> float:
        """TBW."""
        return self._xtol

    @xtol.setter
    def xtol(self, value: float) -> None:
        """TBW."""
        self._xtol = value

    @property
    def memory(self) -> bool:
        """TBW."""
        return self._memory

    @memory.setter
    def memory(self, value: bool) -> None:
        """TBW."""
        self._memory = value

    # SADE #####

    # SGA #####
    @property
    def cr(self) -> float:
        """TBW."""
        return self._cr

    @cr.setter
    def cr(self, value: float) -> None:
        """TBW."""
        if not (0.0 <= value <= 1.0):
            raise ValueError("'cr' must be between 0.0 and 1.0.")

        self._cr = value

    @property
    def eta_c(self) -> float:
        """TBW."""
        return self._eta_c

    @eta_c.setter
    def eta_c(self, value: float) -> None:
        """TBW."""
        self._eta_c = value

    @property
    def m(self) -> float:
        """TBW."""
        return self._m

    @m.setter
    def m(self, value: float) -> None:
        """TBW."""
        if not (0.0 <= value <= 1.0):
            raise ValueError("'m' must be between 0.0 and 1.0.")

        self._m = value

    @property
    def param_m(self) -> float:
        """TBW."""
        return self._param_m

    @param_m.setter
    def param_m(self, value: float) -> None:
        """TBW."""
        self._param_m = value

    @property
    def param_s(self) -> int:
        """TBW."""
        return self._param_s

    @param_s.setter
    def param_s(self, value: int) -> None:
        """TBW."""
        self._param_s = value

    @property
    def crossover(self) -> str:
        """TBW."""
        return self._crossover

    @crossover.setter
    def crossover(self, value: str) -> None:
        """TBW."""
        self._crossover = value

    @property
    def mutation(self) -> str:
        """TBW."""
        return self._mutation

    @mutation.setter
    def mutation(self, value: str) -> None:
        """TBW."""
        self._mutation = value

    @property
    def selection(self) -> str:
        """TBW."""
        return self._selection

    @selection.setter
    def selection(self, value: str) -> None:
        """TBW."""
        self._selection = value

    # SGA #####

    # NLOPT #####
    @property
    def nlopt_solver(self) -> str:
        """TBW."""
        return self._nlopt_solver

    @nlopt_solver.setter
    def nlopt_solver(self, value: str) -> None:
        """TBW."""
        self._nlopt_solver = value

    @property
    def maxtime(self) -> int:
        """TBW."""
        return self._maxtime

    @maxtime.setter
    def maxtime(self, value: int) -> None:
        """TBW."""
        self._maxtime = value

    @property
    def maxeval(self) -> int:
        """TBW."""
        return self._maxeval

    @maxeval.setter
    def maxeval(self, value: int) -> None:
        """TBW."""
        self._maxeval = value

    @property
    def xtol_rel(self) -> float:
        """TBW."""
        return self._xtol_rel

    @xtol_rel.setter
    def xtol_rel(self, value: float) -> None:
        """TBW."""
        self._xtol_rel = value

    @property
    def xtol_abs(self) -> float:
        """TBW."""
        return self._xtol_abs

    @xtol_abs.setter
    def xtol_abs(self, value: float) -> None:
        """TBW."""
        self._xtol_abs = value

    @property
    def ftol_rel(self) -> float:
        """TBW."""
        return self._ftol_rel

    @ftol_rel.setter
    def ftol_rel(self, value: float) -> None:
        """TBW."""
        self._ftol_rel = value

    @property
    def ftol_abs(self) -> float:
        """TBW."""
        return self._ftol_abs

    @ftol_abs.setter
    def ftol_abs(self, value: float) -> None:
        """TBW."""
        self._ftol_abs = value

    @property
    def stopval(self) -> float:
        """TBW."""
        return self._stopval

    @stopval.setter
    def stopval(self, value: float) -> None:
        """TBW."""
        self._stopval = value

    @property
    def local_optimizer(self):
        """TBW."""
        return self._local_optimizer

    @local_optimizer.setter
    def local_optimizer(self, value) -> None:
        """TBW."""
        self._local_optimizer = value

    @property
    def replacement(self) -> str:
        """TBW."""
        return self._replacement

    @replacement.setter
    def replacement(self, value: str) -> None:
        """TBW."""
        self._replacement = value

    @property
    def nlopt_selection(self) -> str:
        """TBW."""
        return self._nlopt_selection

    @nlopt_selection.setter
    def nlopt_selection(self, value: str) -> None:
        """TBW."""
        self._nlopt_selection = value

    # NLOPT #####

    # TODO: This could be refactored for each if-statement
    def get_algorithm(self) -> t.Any:
        """TBW.

        :return:
        """
        if self.type is AlgorithmType.Sade:
            opt_algorithm = pg.sade(
                gen=self.generations,
                variant=self.variant,
                variant_adptv=self.variant_adptv,
                ftol=self.ftol,
                xtol=self.xtol,
                memory=self.memory,
            )
        elif self.type is AlgorithmType.Sga:
            opt_algorithm = pg.sga(
                gen=self.generations,
                cr=self.cr,  # crossover probability
                crossover=self.crossover,  # single, exponential, binomial, sbx
                m=self.m,  # mutation probability
                mutation=self.mutation,  # uniform, gaussian, polynomial
                param_s=self.param_s,  # number of best ind. in 'truncated'/tournament
                selection=self.selection,  # tournament, truncated
                eta_c=self.eta_c,  # distribution index for sbx crossover
                param_m=self.param_m,
            )  # mutation parameter
        elif self.type is AlgorithmType.Nlopt:
            opt_algorithm = pg.nlopt(self.nlopt_solver)
            opt_algorithm.maxtime = (
                self.maxtime
            )  # stop when the optimization time (in seconds) exceeds maxtime
            opt_algorithm.maxeval = (
                self.maxeval
            )  # stop when the number of function evaluations exceeds maxeval
            opt_algorithm.xtol_rel = self.xtol_rel  # relative stopping criterion for x
            opt_algorithm.xtol_abs = self.xtol_abs  # absolute stopping criterion for x
            opt_algorithm.ftol_rel = self.ftol_rel
            opt_algorithm.ftol_abs = self.ftol_abs
            opt_algorithm.stopval = self.stopval
            opt_algorithm.local_optimizer = self.local_optimizer
            opt_algorithm.replacement = self.replacement
            opt_algorithm.selection = self.nlopt_selection
        else:
            raise NotImplementedError

        return opt_algorithm


def to_path_list(values: t.Sequence[t.Union[str, Path]]) -> t.List[Path]:
    """TBW."""
    return [Path(obj) for obj in values]


class Calibration:
    """TBW."""

    def __init__(
        self,
        output_dir: t.Optional[Path] = None,
        fitting: t.Optional[ModelFitting] = None,
        calibration_mode: CalibrationMode = CalibrationMode.Pipeline,
        result_type: ResultType = ResultType.Image,
        result_fit_range: t.Optional[t.List[int]] = None,
        result_input_arguments: t.Optional[list] = None,
        target_data_path: t.Optional[t.List[Path]] = None,
        target_fit_range: t.Optional[t.List[int]] = None,
        fitness_function: t.Optional[ModelFunction] = None,
        algorithm: t.Optional[Algorithm] = None,
        parameters: t.Optional[t.List[ParameterValues]] = None,
        seed: t.Optional[int] = None,
        islands: int = 0,
        weighting_path: t.Optional[list] = None,
    ):
        if seed is not None and seed not in range(100001):
            raise ValueError("'seed' must be between 0 and 100000.")

        if islands not in range(101):
            raise ValueError("'islands' must be between 0 and 100.")

        self._output_dir = output_dir
        self._fitting = fitting
        self._calibration_mode = CalibrationMode(
            calibration_mode
        )  # type: CalibrationMode
        self._result_type = ResultType(result_type)  # type: ResultType
        self._result_fit_range = result_fit_range if result_fit_range else []
        self._result_input_arguments = (
            result_input_arguments if result_input_arguments else []
        )
        self._target_data_path = (
            to_path_list(target_data_path) if target_data_path else []
        )  # type: t.List[Path]
        self._target_fit_range = target_fit_range if target_fit_range else []
        self._fitness_function = fitness_function
        self._algorithm = algorithm
        self._parameters = parameters if parameters else []
        self._seed = np.random.randint(0, 100000) if seed is None else seed  # type: int
        self._islands = islands
        self._weighting_path = weighting_path

    @property
    def output_dir(self) -> Path:
        """TBW."""
        if not self._output_dir:
            raise RuntimeError("No 'output_dir' defined !")

        return self._output_dir

    @output_dir.setter
    def output_dir(self, value: Path) -> None:
        """TBW."""
        self._output_dir = value

    @property
    def fitting(self) -> ModelFitting:
        """TBW."""
        if not self._fitting:
            raise RuntimeError("No 'fitting' defined !")

        return self._fitting

    @fitting.setter
    def fitting(self, value: ModelFitting) -> None:
        """TBW."""
        self._fitting = value

    @property
    def calibration_mode(self) -> CalibrationMode:
        """TBW."""
        return self._calibration_mode

    @calibration_mode.setter
    def calibration_mode(self, value: CalibrationMode) -> None:
        """TBW."""
        self._calibration_mode = value

    @property
    def result_type(self) -> ResultType:
        """TBW."""
        return self._result_type

    @result_type.setter
    def result_type(self, value: ResultType) -> None:
        """TBW."""
        self._result_type = value

    @property
    def result_fit_range(self) -> t.List[int]:
        """TBW."""
        return self._result_fit_range

    @result_fit_range.setter
    def result_fit_range(self, value: list) -> None:
        """TBW."""
        self._result_fit_range = value

    @property
    def result_input_arguments(self) -> list:
        """TBW."""
        return self._result_input_arguments

    @result_input_arguments.setter
    def result_input_arguments(self, value: list) -> None:
        """TBW."""
        self._result_input_arguments = value

    @property
    def target_data_path(self) -> t.List[Path]:
        """TBW."""
        return self._target_data_path

    @target_data_path.setter
    def target_data_path(self, value: t.List[Path]) -> None:
        """TBW."""
        self._target_data_path = value

    @property
    def target_fit_range(self) -> t.List[int]:
        """TBW."""
        return self._target_fit_range

    @target_fit_range.setter
    def target_fit_range(self, value: list) -> None:
        """TBW."""
        self._target_fit_range = value

    @property
    def fitness_function(self) -> ModelFunction:
        """TBW."""
        if not self._fitness_function:
            raise RuntimeError("No 'fitness_function' defined !")

        return self._fitness_function

    @fitness_function.setter
    def fitness_function(self, value: ModelFunction) -> None:
        """TBW."""
        self._fitness_function = value

    @property
    def algorithm(self) -> Algorithm:
        """TBW."""
        if not self._algorithm:
            raise RuntimeError("No 'algorithm' defined !")

        return self._algorithm

    @algorithm.setter
    def algorithm(self, value: Algorithm) -> None:
        """TBW."""
        self._algorithm = value

    @property
    def parameters(self) -> t.List[ParameterValues]:
        """TBW."""
        return self._parameters

    @parameters.setter
    def parameters(self, value: t.List[ParameterValues]) -> None:
        """TBW."""
        self._parameters = value

    @property
    def seed(self) -> int:
        """TBW."""
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        """TBW."""
        if value not in range(100001):
            raise ValueError("'seed' must be between 0 and 100000.")

        self._seed = value

    @property
    def islands(self) -> int:
        """TBW."""
        return self._islands

    @islands.setter
    def islands(self, value: int) -> None:
        """TBW."""
        if value not in range(101):
            raise ValueError("'islands' must be between 0 and 100.")

        self._islands = value

    @property
    def weighting_path(self) -> t.Optional[list]:
        """TBW."""
        return self._weighting_path

    @weighting_path.setter
    def weighting_path(self, value: list) -> None:
        """TBW."""
        self._weighting_path = value

    def run_calibration(self, processor: Processor, output_dir: Path) -> list:
        """TBW.

        :param processor: Processor object
        :param output_dir: Output object
        :return:
        """
        pg.set_global_rng_seed(seed=self.seed)
        logging.info("Seed: %d", self.seed)

        self.output_dir = output_dir
        self.fitting = ModelFitting(processor=processor, variables=self.parameters)

        self.fitting.configure(
            calibration_mode=self.calibration_mode,
            generations=self.algorithm.generations,
            population_size=self.algorithm.population_size,
            simulation_output=self.result_type,
            fitness_func=self.fitness_function,
            target_output=self.target_data_path,
            target_fit_range=self.target_fit_range,
            out_fit_range=self.result_fit_range,
            input_arguments=self.result_input_arguments,
            weighting=self.weighting_path,
            file_path=output_dir,
        )

        prob = pg.problem(self.fitting)
        opt_algorithm = self.algorithm.get_algorithm()
        algo = pg.algorithm(opt_algorithm)

        # try:
        #     bfe = pg.bfe(udbfe=pg.member_bfe())
        # except AttributeError:
        #     bfe = None
        #
        # try:
        #     archi = pg.archipelago(n=self.islands, algo=algo, prob=prob, b=bfe,
        #                            pop_size=self.algorithm.population_size, udi=pg.mp_island())
        # except KeyError:
        #     archi = pg.archipelago(n=self.islands, algo=algo, prob=prob,
        #                            pop_size=self.algorithm.population_size, udi=pg.mp_island())
        #
        # try:
        #     pop = pg.population(prob=prob, size=self.algorithm.population_size, b=bfe)
        # except TypeError:
        #     pop = pg.population(prob=prob, size=self.algorithm.population_size)

        if self.islands > 1:  # default
            archi = pg.archipelago(
                n=self.islands,
                algo=algo,
                prob=prob,
                pop_size=self.algorithm.population_size,
                udi=pg.mp_island(),
            )
            archi.evolve()
            logging.info(archi)
            archi.wait_check()
            champion_f = archi.get_champions_f()
            champion_x = archi.get_champions_x()
        else:
            pop = pg.population(prob=prob, size=self.algorithm.population_size)
            pop = algo.evolve(pop)
            champion_f = [pop.champion_f]
            champion_x = [pop.champion_x]

        self.fitting.file_matching_renaming()
        res = []  # type: list
        for f, x in zip(champion_f, champion_x):
            res += [self.fitting.get_results(overall_fitness=f, parameter=x)]

        logging.info("Calibration ended.")
        return res

    def post_processing(self, calib_results: list, output: Outputs) -> None:
        """TBW."""
        for item in calib_results:
            proc_list = item[0]
            result_dict = item[1]

            output.calibration_outputs(processor_list=proc_list)

            ii = 0
            for processor, target_data in zip(proc_list, self.fitting.all_target_data):
                simulated_data = self.fitting.get_simulated_data(processor)
                output.fitting_plot(
                    target_data=target_data, simulated_data=simulated_data, data_i=ii
                )
                ii += 1
            output.fitting_plot_close(
                result_type=self.result_type, island=result_dict["island"]
            )

        output.calibration_plots(calib_results[0][1])

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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from random import Random
from timeit import default_timer as timer

import dask.array as da
import dask.delayed as delayed
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing_extensions import Literal, Protocol

from pyxel.calibration import (
    AlgorithmType,
    CalibrationMode,
    CalibrationResult,
    Island,
    ResultType,
)
from pyxel.calibration.fitting import ModelFitting, ProblemSingleObjective
from pyxel.parametric.parameter_values import ParameterValues
from pyxel.pipelines import ModelFunction, Processor

if t.TYPE_CHECKING:
    from ..inputs_outputs import CalibrationOutputs

try:
    import pygmo as pg
except ImportError:
    import warnings

    warnings.warn("Cannot import 'pygmo", RuntimeWarning, stacklevel=2)


class IslandProtocol(Protocol):
    """Protocol for a User Define Island."""

    def run_evolve(
        self, algo: "pg.algorithm", pop: "pg.population"
    ) -> t.Tuple["pg.algorithm", "pg.population"]:
        """Run 'evolve' method."""
        ...


class ProblemSerializable:
    """Create a 'problem' with a serializable fitness method.

    Method '.fitness' from a ``pg.problem`` is not serializable with 'cloudpickle'.
    Method ``ProblemSerializable.fitness`` is seriablizable with 'cloudpickle'.

    Examples
    --------
    Create a new pygmo problem and serializable problem
    >>> import pygmo as pg
    >>> prob = pg.problem(...)
    >>> prob_serial = ProblemSerializable(prob)

    Fitness function is working for the pygmo and serializable problem
    >>> import numpy as np
    >>> dvs = np.array([...])
    >>> np.array_equal(prob.fitness(dvs), prob_serial(dvs))
    True

    Serialization is not working for method 'prob.fitness' with 'cloudpickle'
    >>> import cloudpickle
    >>> _ = cloudpickle.dumps(prob.fitness)
    TypeError: cannot pickle 'PyCapsule' object

    Serialization is working with method 'prob_serial.fitness' with 'cloudpickle'
    >>> _ = cloudpickle.dumps(prob_serial.fitness)  # It's working !
    """

    def __init__(self, prob):
        self._prob = prob

    def fitness(self, *args, **kwargs):
        """Compute fitness."""
        return self._prob.fitness(*args, **kwargs)


class AlgoSerializable:
    """Create an 'algorithm' with a serializable evolve method."""

    def __init__(self, algo):
        self._algo = algo

    def evolve(self, *args, **kwargs):
        """Compute 'evolve'."""
        pop = self._algo.evolve(*args, **kwargs)

        return self._algo, pop


def create_archipelago(
    num_islands: int,
    udi: IslandProtocol,
    algo: t.Callable,
    problem: ProblemSingleObjective,
    pop_size: int,
    bfe: t.Optional[t.Callable] = None,
    topology: t.Optional[t.Callable] = None,
    seed: t.Optional[int] = None,
    parallel: bool = True,
    with_bar: bool = False,
) -> "pg.archipelago":
    """Create a new ``archipelago``.

    Parameters
    ----------
    num_islands
    udi
    algo
    problem
    pop_size
    bfe
    topology
    seed
    parallel
    with_bar

    Returns
    -------
    archipelago
        A new archipelago.
    """
    disable_bar = not with_bar  # type: bool
    start_time = timer()  # type: float

    def create_island(seed: t.Optional[int] = None) -> pg.island:
        """Create a new island."""
        return pg.island(
            udi=udi,
            algo=algo,
            prob=problem,
            b=bfe,
            size=pop_size,
            seed=seed,
        )

    if seed is None:
        seeds = [None] * num_islands  # type: t.Sequence[t.Optional[int]]
    else:
        func_rnd = Random()  # type: Random
        func_rnd.seed(seed)
        max_value = np.iinfo(np.uint32).max  # type: int
        seeds = [func_rnd.randint(0, max_value) for _ in range(num_islands)]

    if topology is None:
        topology = pg.topology()

    archi = pg.archipelago(t=topology)

    if parallel:
        with ThreadPoolExecutor(max_workers=num_islands) as executor:
            it = executor.map(create_island, seeds)

            for island in tqdm(
                it, desc="Create islands", total=num_islands, disable=disable_bar
            ):
                archi.push_back(island)
    else:
        it = map(create_island, seeds)
        for island in tqdm(
            it, desc="Create islands", total=num_islands, disable=disable_bar
        ):
            archi.push_back(island)

    stop_time = timer()
    logging.info("Create a new archipelago in %.2f s", stop_time - start_time)

    return archi


def get_logs_from_algo(algo: "pg.algorithm", algo_type: AlgorithmType) -> pd.DataFrame:
    """Get logging information from an algorithm."""
    if algo_type is AlgorithmType.Sade:
        columns = [
            "num_generations",
            "num_evaluations",
            "best_fitness",
            "f",
            "cr",
            "dx",
            "df",
        ]
        algo_to_extract = pg.sade
    else:
        raise NotImplementedError

    algo_extracted = algo.extract(algo_to_extract)
    logs = algo_extracted.get_log()  # type: list

    df = pd.DataFrame(logs, columns=columns)

    return df


def get_logs_from_archi(
    archi: "pg.archipelago", algo_type: AlgorithmType
) -> pd.DataFrame:
    """Get logging information from an archipelago."""
    lst = []
    for id_island, island in enumerate(archi):
        df_island = get_logs_from_algo(algo=island.get_algorithm(), algo_type=algo_type)
        df_island["id_island"] = id_island + 1

        lst.append(df_island)

    df_archipelago = pd.concat(lst)
    return df_archipelago


class DaskBFE:
    """User defined Batch Fitness Evaluator using `Dask`.

    This class is a user-defined batch fitness evaluator based on 'Dask'.
    """

    def __init__(self, chunk_size: t.Optional[int] = None):
        self._chunk_size = chunk_size

    def __call__(self, prob: "pg.problem", dvs_1d: np.ndarray) -> da.Array:
        """Call operator to run the batch fitness evaluator.

        Parameters
        ----------
        prob
        dvs_1d

        Returns
        -------
        array_like
            A 1d array with the fitness parameters.
        """
        ndims_dvs = prob.get_nx()  # type: int
        num_fitness = prob.get_nf()  # type: int

        if self._chunk_size is None:
            chunk_size = max(1, num_fitness // 10)  # type: int
        else:
            chunk_size = self._chunk_size

        # [dvs_1_1, ..., dvs_1_n, dvs_2_1, ..., dvs_2_n, ..., dvs_m_1, ..., dvs_m_n]

        # [[dvs_1_1, ..., dvs_1_n],
        #  [dvs_2_1, ..., dvs_2_n],
        #  ...
        #  [dvs_m_1, ..., dvs_m_n]]

        # Convert 1D Decision Vectors to 2D `dask.Array`
        dvs_2d = da.from_array(
            dvs_1d.reshape((-1, ndims_dvs)),
            chunks=(chunk_size, ndims_dvs),
        )  # type: da.Array

        logging.info("DaskBFE: %i, %i, %r", len(dvs_1d), ndims_dvs, dvs_2d.shape)

        # Create a new problem with a serializable method '.fitness'
        problem_pickable = ProblemSerializable(prob)

        # Create a generalized function to run a 2D input with 'prob.fitness'
        fitness_func = da.gufunc(
            problem_pickable.fitness,
            signature="(i)->(j)",
            output_dtypes=np.float,
            output_sizes={"j": num_fitness},
            vectorize=True,
        )

        fitness_2d = fitness_func(dvs_2d)  # type: da.Array
        fitness_1d = fitness_2d.ravel()  # type: da.Array

        return fitness_1d

    def get_name(self) -> str:
        """Return name of this evaluator."""
        return "Dask batch fitness evaluator"

    def get_extra_info(self) -> str:
        """Return extra information for this evaluator."""
        return f"Dask batch fitness evaluator with chunk_size={self._chunk_size}."


class DaskIsland:
    """User Defined Island usind `Dask`."""

    def run_evolve(
        self, algo: "pg.algorithm", pop: "pg.population"
    ) -> t.Tuple["pg.algorithm", "pg.population"]:
        """Run 'evolve' method from the input `algorithm` to evolve the input `population`.

        Once the evolution is finished, it will return the algorithm used for the
        evolution and the evolved `population`.

        Parameters
        ----------
        algo : pg.algorithm
            Algorithm used to evolve the input population
        pop : pg.population
            Input population.

        Returns
        -------
        tuple of pg.algorithm, pg.population
            The algorithm used for the evolution and the evolved population.
        """
        logging.info("Run evolve %r, %r", pop, algo)

        # Create a new algorithm with a serializable method '.evolve'
        algo_pickable = AlgoSerializable(algo)

        # Run 'algo.evolve' with `Dask`
        new_delayed_pop = delayed(algo_pickable.evolve, nout=2)(
            pop
        )  # type: delayed.Delayed

        (
            new_algo,
            new_pop,
        ) = new_delayed_pop.compute()  # type: t.Tuple[pg.algo, pg.population]

        return new_algo, new_pop

    def get_name(self) -> str:
        """Return Island's name."""
        return "Dask Island"

    # def get_extra_info(self) -> str:
    #     """Return extra information for this Island."""
    #     pass


# TODO: Put classes `Algorithm` and `Calibration` in separated files.
# TODO: Maybe a new class `Sade` could contains some attributes ?
# TODO: Maybe a new class `SGA` could contains some attributes ?
# TODO: Maybe a new class `NLOPT` could contains some attributes ?
class Algorithm:
    """TBW."""

    def __init__(
        self,
        # TODO: Rename 'type' into 'algorithm_type'
        type: Literal["sade", "sga", "nlopt"] = "sade",
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
        local_optimizer: t.Optional["pg.nlopt"] = None,
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
        self._local_optimizer = local_optimizer  # type: t.Optional[pg.nlopt]
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
    def local_optimizer(self) -> t.Optional["pg.nlopt"]:
        """TBW."""
        return self._local_optimizer

    @local_optimizer.setter
    def local_optimizer(self, value: t.Optional["pg.nlopt"]) -> None:
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
    def get_algorithm(self) -> t.Union["pg.sade", "pg.sga", "pg.nlopt"]:
        """TBW.

        :return:
        """
        if self.type is AlgorithmType.Sade:
            sade_algorithm = pg.sade(
                gen=self.generations,
                variant=self.variant,
                variant_adptv=self.variant_adptv,
                ftol=self.ftol,
                xtol=self.xtol,
                memory=self.memory,
            )  # type: pg.sade

            return sade_algorithm

        elif self.type is AlgorithmType.Sga:
            # mutation parameter
            sga_algorithm = pg.sga(
                gen=self.generations,
                cr=self.cr,  # crossover probability
                crossover=self.crossover,  # single, exponential, binomial, sbx
                m=self.m,  # mutation probability
                mutation=self.mutation,  # uniform, gaussian, polynomial
                param_s=self.param_s,  # number of best ind. in 'truncated'/tournament
                selection=self.selection,  # tournament, truncated
                eta_c=self.eta_c,  # distribution index for sbx crossover
                param_m=self.param_m,
            )  # type: pg.sga

            return sga_algorithm

        elif self.type is AlgorithmType.Nlopt:
            opt_algorithm = pg.nlopt(self.nlopt_solver)  # type: pg.nlopt
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

            return opt_algorithm

        else:
            raise NotImplementedError


def to_path_list(values: t.Sequence[t.Union[str, Path]]) -> t.List[Path]:
    """TBW."""
    return [Path(obj).resolve() for obj in values]


class Calibration:
    """TBW."""

    def __init__(
        self,
        outputs: "CalibrationOutputs",
        output_dir: t.Optional[Path] = None,
        fitting: t.Optional[ModelFitting] = None,
        mode: Literal["pipeline", "single_model"] = "pipeline",
        result_type: Literal["image", "signal", "pixel"] = "image",
        result_fit_range: t.Optional[t.Sequence[int]] = None,
        result_input_arguments: t.Optional[t.Sequence[ParameterValues]] = None,
        target_data_path: t.Optional[t.Sequence[Path]] = None,
        target_fit_range: t.Optional[t.Sequence[int]] = None,
        fitness_function: t.Optional[ModelFunction] = None,
        algorithm: t.Optional[Algorithm] = None,
        parameters: t.Optional[t.Sequence[ParameterValues]] = None,
        seed: t.Optional[int] = None,
        num_islands: int = 0,
        num_evolutions: int = 1,
        type_islands: Literal[
            "multiprocessing", "multithreading", "ipyparallel"
        ] = "multiprocessing",
        weighting_path: t.Optional[t.Sequence[Path]] = None,
    ):
        if seed is not None and seed not in range(100001):
            raise ValueError("'seed' must be between 0 and 100000.")

        if num_islands < 1:
            raise ValueError("'num_islands' must superior or equal to 1.")

        self._log = logging.getLogger(__name__)

        self.outputs = outputs

        self._output_dir = output_dir  # type:t.Optional[Path]
        self._fitting = fitting  # type: t.Optional[ModelFitting]

        self._calibration_mode = CalibrationMode(mode)

        self._result_type = ResultType(result_type)  # type: ResultType

        self._result_fit_range = (
            result_fit_range if result_fit_range else []
        )  # type: t.Sequence[int]

        self._result_input_arguments = (
            result_input_arguments if result_input_arguments else []
        )  # type: t.Sequence[ParameterValues]

        self._target_data_path = (
            to_path_list(target_data_path) if target_data_path else []
        )  # type: t.Sequence[Path]
        self._target_fit_range = (
            target_fit_range if target_fit_range else []
        )  # type: t.Sequence[int]

        self._fitness_function = fitness_function  # type: t.Optional[ModelFunction]
        self._algorithm = algorithm  # type: t.Optional[Algorithm]

        self._parameters = (
            parameters if parameters else []
        )  # type: t.Sequence[ParameterValues]

        self._seed = np.random.randint(0, 100000) if seed is None else seed  # type: int

        self._num_islands = num_islands  # type: int
        self._num_evolutions = num_evolutions  # type: int
        self._type_islands = Island(type_islands)  # type:Island

        self._weighting_path = weighting_path  # type: t.Optional[t.Sequence[Path]]

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
    def result_fit_range(self) -> t.Sequence[int]:
        """TBW."""
        return self._result_fit_range

    @result_fit_range.setter
    def result_fit_range(self, value: t.Sequence[int]) -> None:
        """TBW."""
        self._result_fit_range = value

    @property
    def result_input_arguments(self) -> t.Sequence[ParameterValues]:
        """TBW."""
        return self._result_input_arguments

    @result_input_arguments.setter
    def result_input_arguments(self, value: t.Sequence[ParameterValues]) -> None:
        """TBW."""
        self._result_input_arguments = value

    @property
    def target_data_path(self) -> t.Sequence[Path]:
        """TBW."""
        return self._target_data_path

    @target_data_path.setter
    def target_data_path(self, value: t.Sequence[Path]) -> None:
        """TBW."""
        self._target_data_path = value

    @property
    def target_fit_range(self) -> t.Sequence[int]:
        """TBW."""
        return self._target_fit_range

    @target_fit_range.setter
    def target_fit_range(self, value: t.Sequence[int]) -> None:
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
    def parameters(self) -> t.Sequence[ParameterValues]:
        """TBW."""
        return self._parameters

    @parameters.setter
    def parameters(self, value: t.Sequence[ParameterValues]) -> None:
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
    def num_islands(self) -> int:
        """TBW."""
        return self._num_islands

    @num_islands.setter
    def num_islands(self, value: int) -> None:
        """TBW."""
        if value < 1:
            raise ValueError("'num_islands' must superior or equal to 1.")

        self._num_islands = value

    @property
    def num_evolutions(self) -> int:
        """TBW."""
        return self._num_evolutions

    @num_evolutions.setter
    def num_evolutions(self, value: int) -> None:
        """TBW."""
        self._num_evolutions = value

    @property
    def weighting_path(self) -> t.Optional[t.Sequence[Path]]:
        """TBW."""
        return self._weighting_path

    @weighting_path.setter
    def weighting_path(self, value: t.Sequence[Path]) -> None:
        """TBW."""
        self._weighting_path = value

    def run_calibration(
        self, processor: Processor, output_dir: Path, with_bar: bool = True
    ) -> t.Tuple[t.Sequence[CalibrationResult], pd.DataFrame]:
        """TBW.

        Parameters
        ----------
        processor
        output_dir

        Returns
        -------
        Sequence of `CalibrationResult`
        """
        pg.set_global_rng_seed(seed=self.seed)
        self._log.info("Seed: %d", self.seed)

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

        # Create a Pygmo problem
        prob = pg.problem(self.fitting)  # type: pg.problem
        self._log.info(prob)

        total_num_generations = (
            self._num_evolutions * self.algorithm.generations
        )  # type: int

        # Create a Pygmo algorithm
        algo = pg.algorithm(self.algorithm.get_algorithm())  # type: pg.algorithm
        self._log.info(algo)

        if self.num_islands > 1:  # default
            # Create an archipelago
            user_defined_island = DaskIsland()
            user_defined_bfe = DaskBFE()

            verbosity_level = max(1, self.algorithm.population_size // 100)  # type: int
            algo.set_verbosity(verbosity_level)

            archi = create_archipelago(
                num_islands=self.num_islands,
                udi=user_defined_island,
                algo=algo,
                problem=prob,
                pop_size=self.algorithm.population_size,
                bfe=user_defined_bfe,
                seed=self.seed,
                with_bar=with_bar,
            )  # type: pg.archipelago

            # TODO: Missing parameter 't' for a user-defined topology
            # archi = pg.archipelago(
            #    n=self.num_islands,
            #    udi=user_defined_island,
            #                algo=algo,
            #                prob=prob,
            #                pop_size=self.algorithm.population_size,
            #                b=user_defined_bfe,
            #             )

            # TODO: Remove this
            pd.set_option("display.max_columns", 10)
            pd.set_option("display.width", 1000)
            pd.set_option("display.max_rows", 30)

            df_all_logs = pd.DataFrame()

            bars = {}
            for idx in range(self.num_islands):
                bars[idx + 1] = tqdm(
                    total=int(total_num_generations),
                    position=idx,
                    desc=f"Island {idx+1:02d}",
                    unit="generations",
                )

            for id_evolution in range(self._num_evolutions):
                # Call all 'evolve()' methods on all islands
                archi.evolve()
                self._log.info(archi)

                # Block until all evolutions have finished and raise the first exception
                # that was encountered
                archi.wait_check()

                df_logs = get_logs_from_archi(
                    archi=archi, algo_type=self.algorithm.type
                )
                df_logs["id_evolution"] = id_evolution + 1

                df_all_logs = df_all_logs.append(df_logs)

                df_last = df_all_logs.groupby(["id_island"]).last()
                df_last = df_last.assign(
                    global_num_generations=df_last["num_generations"]
                    * df_last["id_evolution"]
                )

                for id_island, serie in df_last.iterrows():
                    num_generations = int(serie["global_num_generations"])
                    bars[id_island].update(num_generations)

            # Get fitness and decision vectors of the num_islands' champions
            champions_1d_fitness = archi.get_champions_f()  # type: t.List[np.ndarray]
            champions_1d_decision = archi.get_champions_x()  # type: t.List[np.ndarray]
        else:
            self._log.info("Initialize optimization algorithm")
            pop = pg.population(prob=prob, size=self.algorithm.population_size)

            self._log.info("Start optimization algorithm")

            pop = algo.evolve(pop)

            # Get log information
            df_all_logs = get_logs_from_algo(algo=algo, algo_type=self.algorithm.type)

            # Get fitness and decision vector of the population champion
            champions_1d_fitness = [pop.champion_f]
            champions_1d_decision = [pop.champion_x]

        df_all_logs = df_all_logs.reset_index(drop=True).assign(
            global_num_generations=lambda df: (df["id_evolution"] - 1)
            * self.algorithm.generations
            + df["num_generations"]
        )

        self.fitting.file_matching_renaming()
        res = []  # type: t.List[CalibrationResult]

        for f, x in zip(champions_1d_fitness, champions_1d_decision):
            res += [self.fitting.get_results(overall_fitness=f, parameter=x)]

        self._log.info("Calibration ended.")
        return res, df_all_logs

    # TODO: Speed-up this function
    def post_processing(
        self,
        calib_results: t.Sequence[CalibrationResult],
        output: "CalibrationOutputs",
    ) -> None:
        """TBW."""
        for one_calib_result in tqdm(calib_results):  # type: CalibrationResult

            # TODO: Create a new method in output called '.save_processor(processor)'
            output.calibration_outputs(processor_list=one_calib_result.processors)

            for idx, (processor, target_data) in enumerate(
                zip(one_calib_result.processors, self.fitting.all_target_data)
            ):
                simulated_data = self.fitting.get_simulated_data(processor)
                output.fitting_plot(
                    target_data=target_data, simulated_data=simulated_data, data_i=idx
                )

            output.fitting_plot_close(
                result_type=self.result_type, island=one_calib_result.island
            )

        first_calib_result = calib_results[0]  # type: CalibrationResult
        output.calibration_plots(
            results=first_calib_result.results, fitness=first_calib_result.fitness
        )

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
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing_extensions import Literal

from pyxel.calibration import (
    Algorithm,
    CalibrationMode,
    CalibrationResult,
    DaskBFE,
    DaskIsland,
    Island,
    ResultType,
    create_archipelago,
    get_logs_from_algo,
    get_logs_from_archi,
)
from pyxel.calibration.fitting import ModelFitting
from pyxel.parametric.parameter_values import ParameterValues
from pyxel.pipelines import ModelFunction, Processor

if t.TYPE_CHECKING:
    from ..inputs_outputs import CalibrationOutputs

try:
    import pygmo as pg
except ImportError:
    import warnings

    warnings.warn("Cannot import 'pygmo", RuntimeWarning, stacklevel=2)


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
        topology: t.Literal["unconnected", "ring", "fully_connected"] = "unconnected",
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
        self._topology = (
            topology
        )  # type: t.Literal['unconnected', 'ring', 'fully_connected']

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
    def topology(self) -> t.Literal["unconnected", "ring", "fully_connected"]:
        return self._topology

    @topology.setter
    def topology(self, value: t.Any) -> None:
        if value not in ["unconnected", "ring", "fully_connected"]:
            raise ValueError(
                "Expecting value: 'unconnected', 'ring' or 'fully_connected'"
            )

        self._topology = value

    @property
    def weighting_path(self) -> t.Optional[t.Sequence[Path]]:
        """TBW."""
        return self._weighting_path

    @weighting_path.setter
    def weighting_path(self, value: t.Sequence[Path]) -> None:
        """TBW."""
        self._weighting_path = value

    def run_calibration(
        self, processor: Processor, output_dir: Path, with_progress_bar: bool = True
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

        total_num_generations = self._num_evolutions * self.algorithm.generations

        # Create a Pygmo algorithm
        algo = pg.algorithm(self.algorithm.get_algorithm())  # type: pg.algorithm
        self._log.info(algo)

        if self.num_islands > 1:  # default
            # Create an archipelago
            user_defined_island = DaskIsland()
            user_defined_bfe = DaskBFE()

            verbosity_level = max(1, self.algorithm.population_size // 100)  # type: int
            algo.set_verbosity(verbosity_level)

            if self.topology == "unconnected":
                topo = pg.unconnected()
            elif self.topology == "ring":
                topo = pg.ring()
            elif self.topology == "fully_connected":
                topo = pg.fully_connected()
            else:
                raise NotImplementedError(f"topology {self.topology!r}")

            archi = create_archipelago(
                num_islands=self.num_islands,
                udi=user_defined_island,
                algo=algo,
                problem=prob,
                pop_size=self.algorithm.population_size,
                bfe=user_defined_bfe,
                topology=topo,
                seed=self.seed,
                with_bar=with_progress_bar,
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

            df_all_logs = pd.DataFrame()

            # Create progress bars
            max_num_progress_bars = 10
            num_progress_bars = min(self.num_islands, max_num_progress_bars)
            num_islands_per_bar = math.ceil(self.num_islands // num_progress_bars)

            progress_bars = []
            if with_progress_bar:
                for idx in range(num_progress_bars):
                    if num_islands_per_bar == 1:
                        desc = f"Island {idx+1:02d}"
                    else:
                        first_island = idx * num_islands_per_bar + 1
                        last_island = (idx + 1) * num_islands_per_bar + 1

                        if last_island > self.num_islands:
                            last_island = self.num_islands

                        desc = f"Islands {first_island:02d}-{last_island:02d}"

                    new_bar = tqdm(
                        total=int(total_num_generations),
                        position=idx,
                        desc=desc,
                        unit=" generations",
                    )

                    progress_bars.append(new_bar)

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
                df_logs = df_logs.assign(
                    id_evolution=id_evolution + 1,
                    id_progress_bar=lambda df: (df["id_island"] // num_islands_per_bar),
                )

                df_all_logs = df_all_logs.append(df_logs)

                if with_progress_bar:
                    df_last = df_all_logs.groupby("id_progress_bar").last()
                    df_last = df_last.assign(
                        global_num_generations=df_last["num_generations"]
                        * df_last["id_evolution"]
                    )

                    for id_progress_bar, serie in df_last.iterrows():
                        num_generations = int(serie["global_num_generations"])
                        # print(f'{id_evolution=}, {id_progress_bar=}, {num_generations=}')
                        progress_bars[id_progress_bar - 1].update(num_generations)

            for progress_bar in progress_bars:
                progress_bar.close()
                del progress_bar

            t0 = timer()
            # Get fitness and decision vectors of the num_islands' champions
            champions_1d_fitness = archi.get_champions_f()  # type: t.List[np.ndarray]
            champions_1d_decision = archi.get_champions_x()  # type: t.List[np.ndarray]

            t1 = timer()
            print(f"Get fitness in {t1-t0:.2f} s")

        else:
            # self._log.info("Initialize optimization algorithm")
            # pop = pg.population(prob=prob, size=self.algorithm.population_size)
            #
            # self._log.info("Start optimization algorithm")
            #
            # pop = algo.evolve(pop)
            #
            # # Get log information
            # df_all_logs = get_logs_from_algo(algo=algo, algo_type=self.algorithm.type)
            #
            # # Get fitness and decision vector of the population champion
            # champions_1d_fitness = [pop.champion_f]
            # champions_1d_decision = [pop.champion_x]
            raise NotImplementedError("Not implemented for 1 island.")

        df_all_logs = df_all_logs.reset_index(drop=True).assign(
            global_num_generations=lambda df: (df["id_evolution"] - 1)
            * self.algorithm.generations
            + df["num_generations"]
        )

        t0 = timer()
        self.fitting.file_matching_renaming()
        t1 = timer()
        print(f"File matching renaming in {t1-t0:.2f} s")

        res = []  # type: t.List[CalibrationResult]

        for f, x in tqdm(
            zip(champions_1d_fitness, champions_1d_decision),
            desc="Get results",
            disable=not with_progress_bar,
        ):
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

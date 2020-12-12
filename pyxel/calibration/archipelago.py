#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Sub-package to create 'archipelagos'."""
import logging
import math
import typing as t
from concurrent.futures.thread import ThreadPoolExecutor
from random import Random
from timeit import default_timer as timer

import dask.array as da
import numpy as np
import pandas as pd
import pygmo as pg
import xarray as xr
from dask.delayed import Delayed
from tqdm.auto import tqdm

from pyxel.calibration import Algorithm, AlgorithmType, IslandProtocol
from pyxel.calibration.fitting import ModelFitting


class ProgressBars:
    """TBW."""

    def __init__(
        self,
        num_islands: int,
        total_num_generations: int,
        max_num_progress_bars: int = 10,
    ):
        self._num_progress_bars = min(num_islands, max_num_progress_bars)
        self._num_islands_per_bar = math.ceil(num_islands // self._num_progress_bars)

        self._progress_bars = []
        for idx in range(self._num_progress_bars):
            if self._num_islands_per_bar == 1:
                desc = f"Island {idx + 1:02d}"
            else:
                first_island = idx * self._num_islands_per_bar + 1
                last_island = (idx + 1) * self._num_islands_per_bar + 1

                if last_island > num_islands:
                    last_island = num_islands

                desc = f"Islands {first_island:02d}-{last_island:02d}"

            # Create a new bar
            new_bar = tqdm(
                total=int(total_num_generations),
                position=idx,
                desc=desc,
                unit=" generations",
            )

            self._progress_bars.append(new_bar)

    # TODO: Simplify this method
    def update(self, df: pd.DataFrame) -> None:
        """TBW."""
        df = df.assign(
            id_progress_bar=lambda df: (df["id_island"] // self._num_islands_per_bar)
        )
        df_last = df.groupby("id_progress_bar").last()
        df_last = df_last.assign(
            global_num_generations=df_last["num_generations"] * df_last["id_evolution"]
        )

        for id_progress_bar, serie in df_last.iterrows():
            num_generations = int(serie["global_num_generations"])
            # print(f'{id_evolution=}, {id_progress_bar=}, {num_generations=}')
            self._progress_bars[id_progress_bar - 1].update(num_generations)

    def close(self) -> None:
        """TBW."""
        for progress_bar in self._progress_bars:
            progress_bar.close()
            del progress_bar


class ArchipelagoLogs:
    """TBW."""

    def __init__(self, algo: Algorithm, num_generations: int):
        self._df = pd.DataFrame()
        self._algo_type = algo.type  # type: AlgorithmType
        self._num_generations = num_generations

    def _from_algo(self, algo: "pg.algorithm") -> pd.DataFrame:
        """Get logging information from an algorithm."""
        if self._algo_type is AlgorithmType.Sade:
            columns = (
                "num_generations",
                "num_evaluations",
                "best_fitness",
                "f",
                "cr",
                "dx",
                "df",
            )
            algo_to_extract = pg.sade
        else:
            raise NotImplementedError

        algo_extracted = algo.extract(algo_to_extract)
        logs = algo_extracted.get_log()  # type: list

        df = pd.DataFrame(logs, columns=columns)

        return df

    def _from_archi(self, archi: "pg.archipelago") -> pd.DataFrame:
        """Get logging information from an archipelago."""
        lst = []
        for id_island, island in enumerate(archi):
            df_island = self._from_algo(algo=island.get_algorithm())
            df_island["id_island"] = id_island + 1

            lst.append(df_island)

        df_archipelago = pd.concat(lst)
        return df_archipelago

    def append(self, archi: pg.archipelago, id_evolution: int) -> None:
        """TBW."""
        partial_df = self._from_archi(archi=archi)
        partial_df["id_evolution"] = id_evolution

        self._df = self._df.append(partial_df)

    def get_total(self) -> pd.DataFrame:
        """TBW."""
        new_df = self._df.reset_index(drop=True)
        new_df["global_num_generations"] = (
            new_df["id_evolution"] * self._num_generations * new_df["num_generations"]
        )

        return new_df


def extract_data_2d(df_processors: pd.DataFrame, rows: int, cols: int) -> xr.Dataset:
    lst = []
    for _, row in df_processors.iterrows():
        island = row["island"]  # type: int
        id_processor = row["id_processor"]  # type: int
        processor = row["processor"]  # type: Delayed

        image_delayed = processor.detector.image.array  # type: Delayed
        signal_delayed = processor.detector.signal.array  # type: Delayed
        pixel_delayed = processor.detector.pixel.array  # type: Delayed

        image_2d = da.from_delayed(
            image_delayed, shape=(rows, cols), dtype=np.float
        )  # type: da.Array
        signal_2d = da.from_delayed(
            signal_delayed, shape=(rows, cols), dtype=np.float
        )  # type: da.Array
        pixel_2d = da.from_delayed(
            pixel_delayed, shape=(rows, cols), dtype=np.float
        )  # type: da.Array

        partial_ds = xr.Dataset()
        partial_ds["simulated_image"] = xr.DataArray(image_2d, dims=["y", "x"])
        partial_ds["simulated_signal"] = xr.DataArray(signal_2d, dims=["y", "x"])
        partial_ds["simulated_pixel"] = xr.DataArray(pixel_2d, dims=["y", "x"])

        lst.append(
            partial_ds.assign_coords(
                coords={"island": island, "id_processor": id_processor}
            ).expand_dims(["island", "id_processor"])
        )

    ds = xr.combine_by_coords(lst).assign_coords(
        coords={"y": range(rows), "x": range(cols)}
    )

    return ds


class MyArchipelago:
    def __init__(
        self,
        num_islands: int,
        udi: IslandProtocol,
        algorithm: Algorithm,
        problem: ModelFitting,
        pop_size: int,
        bfe: t.Optional[t.Callable] = None,
        topology: t.Optional[t.Callable] = None,
        seed: t.Optional[int] = None,
        parallel: bool = True,
        with_bar: bool = False,
    ):
        self._log = logging.getLogger(__name__)

        self.num_islands = num_islands
        self.udi = udi
        self.algorithm = algorithm
        self.problem = problem
        self.pop_size = pop_size
        self.bfe = bfe
        self.topology = topology
        self.seed = seed
        self.parallel = parallel
        self.with_bar = with_bar

        # Create a Pygmo archipelago
        self._archi = pg.archipelago(t=self.topology)

        # Create a Pygmo algorithm
        verbosity_level = max(1, self.algorithm.population_size // 100)  # type: int

        self._algo = pg.algorithm(self.algorithm.get_algorithm())  # type: pg.algorithm
        self._algo.set_verbosity(verbosity_level)
        self._log.info(self._algo)

        # Create a Pygmo problem
        self._prob = pg.problem(self.problem)  # type: pg.problem
        self._log.info(self._prob)

    def create(self):
        disable_bar = not self.with_bar  # type: bool
        start_time = timer()  # type: float

        def create_island(seed: t.Optional[int] = None) -> pg.island:
            """Create a new island."""
            return pg.island(
                udi=self.udi,
                algo=self._algo,
                prob=self._prob,
                b=self.bfe,
                size=self.pop_size,
                seed=seed,
            )

        if self.seed is None:
            seeds = [None] * self.num_islands  # type: t.Sequence[t.Optional[int]]
        else:
            func_rnd = Random()  # type: Random
            func_rnd.seed(self.seed)
            max_value = np.iinfo(np.uint32).max  # type: int
            seeds = [func_rnd.randint(0, max_value) for _ in range(self.num_islands)]

        if self.parallel:
            with ThreadPoolExecutor(max_workers=self.num_islands) as executor:
                it = executor.map(create_island, seeds)

                for island in tqdm(
                    it,
                    desc="Create islands",
                    total=self.num_islands,
                    disable=disable_bar,
                ):
                    self._archi.push_back(island)
        else:
            it = map(create_island, seeds)
            for island in tqdm(
                it, desc="Create islands", total=self.num_islands, disable=disable_bar
            ):
                self._archi.push_back(island)

        stop_time = timer()
        logging.info("Create a new archipelago in %.2f s", stop_time - start_time)

    def run_evolve(
        self, num_evolutions: int
    ) -> t.Tuple[xr.Dataset, pd.DataFrame, pd.DataFrame]:
        logs = ArchipelagoLogs(
            self.algorithm, num_generations=self.algorithm.generations
        )

        total_num_generations = num_evolutions * self.algorithm.generations

        progress_bars = None  # type: t.Optional[ProgressBars]
        if self.with_bar:
            # Create progress bar(s)
            progress_bars = ProgressBars(
                num_islands=self.num_islands,
                total_num_generations=total_num_generations,
            )

        # TODO: Refactor this
        # Run the archipelago with several evolutions
        for id_evolution in range(num_evolutions):
            # Call all 'evolve()' methods on all islands
            self._archi.evolve()
            self._log.info(self._archi)

            # Block until all evolutions have finished and raise the first exception
            # that was encountered
            self._archi.wait_check()

            logs.append(archi=self._archi, id_evolution=id_evolution + 1)

            if progress_bars:
                progress_bars.update(logs.get_total())

        if progress_bars:
            progress_bars.close()

        # Get logging information
        df_all_logs = logs.get_total()  # type: pd.DataFrame

        # Get fitness and decision vectors of the num_islands' champions
        champions_1d_fitness = self._archi.get_champions_f()  # type: t.List[np.array]
        champions_1d_decision = self._archi.get_champions_x()  # type: t.List[np.array]

        # Get the champions as a Dataset
        champions = xr.Dataset()
        champions["champion_fitness"] = xr.DataArray(
            np.ravel(champions_1d_fitness), dims="island"
        )
        champions["champion_decision"] = xr.DataArray(
            champions_1d_decision, dims=["island", "param_id"]
        )
        champions["champion_parameters"] = self.problem.update_parameter(
            champions["champion_decision"]
        )

        # Get processor(s)
        df_processors = self.problem.apply_parameters_to_processors(
            parameters=champions["champion_parameters"],
        )

        assert isinstance(self.problem.sim_fit_range, tuple)
        slice_rows, slice_cols = self.problem.sim_fit_range

        geometry = self.problem.processor.detector.geometry

        all_simulated_full = extract_data_2d(
            df_processors=df_processors,
            rows=geometry.row,
            cols=geometry.col,
        )

        all_data_fit_range = all_simulated_full.sel(y=slice_rows, x=slice_cols)
        all_data_fit_range["target"] = xr.DataArray(
            self.problem.all_target_data, dims=["id_processor", "y", "x"]
        )

        ds = xr.merge([champions, all_data_fit_range])  # type: xr.Dataset

        return ds, df_processors, df_all_logs

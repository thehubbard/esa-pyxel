#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Sub-package to create 'archipelagos'."""
import logging
import typing as t
from concurrent.futures.thread import ThreadPoolExecutor
from random import Random
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import pygmo as pg
from tqdm.auto import tqdm

from pyxel.calibration import (
    AlgorithmType,
    IslandProtocol,
    ProblemSingleObjective,
    get_logs_from_algo,
)

__all__ = ["get_logs_from_archi", "create_archipelago"]


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

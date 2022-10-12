#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
"""Util functions to handle random seeds."""

from contextlib import contextmanager
from functools import wraps
from typing import Callable

import numpy as np


@contextmanager
def change_random_state(seed: int):
    """TBW.

    Parameters
    ----------
    seed: int
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def temporary_random_state(func: Callable) -> Callable:
    """Temporarily change numpy random seed within a function.

    Parameters
    ----------
    func: callable

    Returns
    -------
    inner: callable
    """

    @wraps(func)
    def inner(*args, seed=None, **kwargs):
        if seed is None:
            return func(*args, seed=seed, **kwargs)
        else:
            with change_random_state(seed):
                return func(*args, seed=seed, **kwargs)

    return inner

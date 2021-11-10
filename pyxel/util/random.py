#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
"""Util functions to handle random seeds."""

import typing as t
from contextlib import contextmanager

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


def temporary_random_state(func: t.Callable) -> t.Callable:
    """Temporarily change numpy random seed within a function.

    Parameters
    ----------
    func: callable

    Returns
    -------
    inner: callable
    """

    def inner(*args, seed=None, **kwargs):
        if seed is not None:
            with change_random_state(seed):
                return func(*args, seed=seed, **kwargs)
        else:
            seed = np.random.randint(10000)
            return func(*args, seed=seed, **kwargs)

    return inner

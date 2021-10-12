#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
"""TBW."""

import typing as t

import numpy as np

from pyxel.evaluator import eval_range
from pyxel.inputs.loader import load_table


class Sampling:
    """TBW."""

    def __init__(
        self,
        readout_times: t.Optional[t.Union[t.Sequence, str]] = None,
        readout_times_from_file: t.Optional[str] = None,
        start_time: float = 0.0,
        non_destructive_readout: bool = False,
    ):
        """Create an instance of Sampling class.

        Parameters
        ----------
        outputs
        times
        times_from_file
        start_time
        non_destructive_readout
        """
        self._time_domain_simulation = True

        if readout_times is not None and readout_times_from_file is not None:
            raise ValueError("Both times and times_from_file specified. Choose one.")
        elif readout_times is None and readout_times_from_file is None:
            self._times = np.array(
                [1]
            )  # by convention default sampling/exposure time is 1 second
            self._time_domain_simulation = False
        elif readout_times_from_file:
            self._times = (
                load_table(readout_times_from_file).to_numpy(dtype=float).flatten()
            )
        elif readout_times:
            self._times = np.array(eval_range(readout_times), dtype=float)
        else:
            raise ValueError("Sampling times not specified.")

        self._non_destructive_readout = non_destructive_readout

        self._times_linear = True  # type: bool
        self._start_time = start_time  # type:float
        self._steps = np.array([])  # type: np.ndarray
        self._num_steps = 0  # type: int

        self._set_steps()

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__  # type: str
        return f"{cls_name}<num_steps={self._num_steps}>"

    def _set_steps(self) -> None:
        """TBW."""
        self._times, self._steps = calculate_steps(self._times, self._start_time)
        self._times_linear = bool(np.all(self._steps == self._steps[0]))
        self._num_steps = len(self._times)

    def time_step_it(self) -> t.Iterator[t.Tuple[float, float]]:
        """TBW."""
        return zip(self._times, self._steps)

    @property
    def times(self) -> t.Any:
        """TBW."""
        return self._times

    @property
    def time_domain_simulation(self) -> bool:
        """TBW."""
        return self._time_domain_simulation

    @property
    def steps(self) -> np.ndarray:
        """TBW."""
        return self._steps

    @property
    def non_destructive_readout(self) -> bool:
        """TBW."""
        return self._non_destructive_readout


def calculate_steps(
    times: np.ndarray, start_time: float
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Calculate time differences for a given array and start time.

    Parameters
    ----------
    times: ndarray
    start_time: float

    Returns
    -------
    times: ndarray
        Modified times according to start time.
    steps: ndarray
        Steps corresponding to times.
    """
    if start_time == times[0]:
        steps = np.diff(times, axis=0)
        times = times[1:]
    else:
        steps = np.diff(
            np.concatenate((np.array([start_time]), times), axis=0),
            axis=0,
        )
    return times, steps

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
"""Dynamic detector properties class."""


class DynamicProperties:
    """Dynamic detector properties."""

    def __init__(
        self,
        num_steps: int,
        start_time: float = 0.0,
        end_time: float = 0.0,
        ndreadout: bool = False,
        linear: bool = True,
    ):
        self._start_time = start_time  # type: float
        self._end_time = end_time  # type: float
        self._time_step = 0.0  # type: float
        self._time = 0.0  # type: float
        self._non_destructive = ndreadout  # type: bool
        self._read_out = True  # type: bool
        self._pipeline_count = 0  # type: int
        self._num_steps = num_steps  # type: int
        self._times_linear = linear  # type: bool

    @property
    def time(self) -> float:
        """TBW."""
        return self._time

    @time.setter
    def time(self, value: float) -> None:
        """TBW."""
        self._time = value

    @property
    def non_destructive_readout(self) -> bool:
        """TBW."""
        return self._non_destructive

    @property
    def time_step(self) -> float:
        """TBW."""
        return self._time_step

    @time_step.setter
    def time_step(self, value: float) -> None:
        """TBW."""
        self._time_step = value

    @property
    def num_steps(self) -> int:
        """TBW."""
        return self._num_steps

    @property
    def times_linear(self) -> bool:
        """TBW."""
        return self._times_linear

    @property
    def pipeline_count(self) -> float:
        """TBW."""
        return self._pipeline_count

    @pipeline_count.setter
    def pipeline_count(self, value: int) -> None:
        """TBW."""
        self._pipeline_count = value

    @property
    def read_out(self) -> bool:
        """TBW."""
        return self._read_out

    @read_out.setter
    def read_out(self, value: bool) -> None:
        """TBW."""
        self._read_out = value

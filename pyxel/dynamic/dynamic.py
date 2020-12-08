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

if t.TYPE_CHECKING:
    from ..inputs_outputs import DynamicOutputs


class Dynamic:
    """TBW."""

    def __init__(
        self,
        outputs: "DynamicOutputs",
        t_step: float,
        steps: int,
        non_destructive_readout: bool = False,
    ):
        self.outputs = outputs
        self._t_step = t_step
        self._steps = steps
        self._non_destructive_readout = non_destructive_readout

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__  # type: str
        return f"{cls_name}<outputs={self.outputs!r}>"

    @property
    def t_step(self):
        """TBW."""
        return self._t_step

    @property
    def steps(self):
        """TBW."""
        return self._steps

    @property
    def non_destructive_readout(self):
        """TBW."""
        return self._non_destructive_readout

    @non_destructive_readout.setter
    def non_destructive_readout(self, non_destructive_readout: bool) -> None:
        """TBW."""
        self._non_destructive_readout = non_destructive_readout

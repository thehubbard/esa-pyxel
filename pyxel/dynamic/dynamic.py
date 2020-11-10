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
    from ..util import Outputs


class Dynamic:
    """TBW."""

    def __init__(
        self, outputs: Outputs, non_destructive_readout: bool, t_step: float, steps: int
    ):
        self.outputs = outputs
        self.non_destructive_readout = non_destructive_readout
        self.t_step = t_step
        self.steps = steps

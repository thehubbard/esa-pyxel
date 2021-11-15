#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import numpy as np
import pytest

from pyxel.models.readout_electronics import dead_time
from pyxel.models.readout_electronics.dead_time import apply_dead_time_filter


def test_apply_dead_time_filter():
    """Test function 'apply_dead_time_filter."""
    data_2d = np.array(
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]
    )

    result_2d = apply_dead_time_filter(phase_2d=data_2d, maximum_count=12)

    exp_data_2d = np.array(
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 12, 12], [12, 12, 12, 12, 12]]
    )

    np.testing.assert_equal(data_2d, exp_data_2d)

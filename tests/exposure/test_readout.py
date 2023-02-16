#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyxel.exposure import Readout


@pytest.mark.parametrize(
    "times, exp_times",
    [
        ([0.1], np.array([0.1])),
        ([1, 2, 3], np.array([1.0, 2.0, 3.0])),
        ("numpy.arange(0.5, 2, 0.5)", np.array([0.5, 1.0, 1.5])),
    ],
)
def test_readout_times(times, exp_times):
    """Test 'Readout' with 'times' parameters."""
    obj = Readout(times=times)

    assert_array_equal(obj.times, exp_times)


@pytest.mark.parametrize(
    "times",
    [[3, 2, 1], [1, 2, 3, 3, 4], [1, 3, 2, 4], "numpy.linspace(start=10, stop=1)"],
)
def test_readout_times_not_monotonic(times):
    """Test 'Readout' with value non-monotonic increasing."""
    with pytest.raises(ValueError, match=r"Readout times must be strictly increasing"):
        _ = Readout(times)


def test_readout_times_error_non_zero():
    """Test 'Readout' with times starting with 0."""
    with pytest.raises(ValueError, match=r"Readout times should be non-zero values"):
        _ = Readout(times=[0])


@pytest.mark.parametrize(
    "times, start_time",
    [
        ([1], 1),
        ([1], 2),
    ],
)
def test_readout_times_greater_than_start(times, start_time):
    """Test 'Readout' with times greater than start time."""
    with pytest.raises(
        ValueError, match=r"Readout times should be greater than start time"
    ):
        _ = Readout(times=times, start_time=start_time)

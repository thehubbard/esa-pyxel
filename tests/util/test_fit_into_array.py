#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
import numpy as np
import pytest
import typing as t
from pyxel.util import fit_into_array


@pytest.mark.parametrize(
    "output_shape, relative_position, align, allow_smaller_array, exp_exc, exp_error",
    [
        pytest.param(
            (10, 10),
            (0, 0),
            "foo",
            True,
            ValueError,
            r"'.*' is not a valid Alignment",
        ),
        pytest.param(
            (20, 20),
            (0, 0),
            None,
            False,
            ValueError,
            "Input array too small to fit into the desired shape!.",
        ),
        pytest.param(
            (10, 10),
            (15, 15),
            None,
            True,
            ValueError,
            "No overlap of array and target in Y and X dimension.",
        ),
        pytest.param(
            (10, 10),
            (15, 0),
            None,
            True,
            ValueError,
            "No overlap of array and target in Y dimension.",
        ),
        pytest.param(
            (10, 10),
            (0, 15),
            None,
            True,
            ValueError,
            "No overlap of array and target in X dimension.",
        ),
    ],
)
def test_fit_into_array_bad_inputs(
    output_shape: t.Tuple[int, int],
    relative_position: t.Tuple[int, int],
    align: t.Optional[
        t.Literal["center", "top_left", "top_right", "bottom_left", "bottom_right"]
    ],
    allow_smaller_array: bool,
    exp_exc,
    exp_error,
):
    """Test function 'fit_into_array' with bad inputs."""

    array = np.ones((10, 10))

    with pytest.raises(exp_exc, match=exp_error):
        fit_into_array(
            array=array,
            output_shape=output_shape,
            relative_position=relative_position,
            align=align,
            allow_smaller_array=allow_smaller_array,
        )

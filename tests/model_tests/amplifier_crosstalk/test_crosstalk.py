#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
# import pytest
# import numpy as np
# from pyxel.models.readout_electronics.amplifier_crosstalk import get_channel_slices

# @pytest.fixture()
# def shape():
#     return (100, 100)
#
# @pytest.mark.parametrize("channel_matrix,expected", [
#     (np.array([1, 2]), [[slice(0, 50, None), slice(0, 100, None)], [slice(50, 100, None), slice(0, 100, None)]]),
# ])
# def test_channel_slices(shape, channel_matrix, expected):
#     assert get_channel_slices(shape, channel_matrix) == expected
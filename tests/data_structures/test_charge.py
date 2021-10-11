#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import pandas as pd
import pytest

from pyxel.data_structure import Charge


@pytest.fixture
def empty_charge() -> Charge:
    """Create an empty `Charge` object."""
    return Charge()


def test_empty_charge(empty_charge: Charge):
    """Test an empty `Charge` object."""
    assert isinstance(empty_charge, Charge)

    assert empty_charge.nextid == 0
    pd.testing.assert_frame_equal(
        empty_charge.frame,
        pd.DataFrame(
            data=None,
            columns=(
                "charge",
                "number",
                "init_energy",
                "energy",
                "init_pos_ver",
                "init_pos_hor",
                "init_pos_z",
                "position_ver",
                "position_hor",
                "position_z",
                "velocity_ver",
                "velocity_hor",
                "velocity_z",
            ),
            dtype=float,
        ),
    )

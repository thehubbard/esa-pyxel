#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import typing as t

import pytest

from pyxel.models.optics.poppy import (
    CircularAperture,
    HexagonAperture,
    MultiHexagonalAperture,
    RectangleAperture,
    SecondaryObscuration,
    SineWaveWFE,
    SquareAperture,
    ThinLens,
    ZernikeWFE,
    create_optical_parameter,
)


@pytest.mark.parametrize(
    "dct, exp_parameter",
    [
        pytest.param({}, None, marks=pytest.mark.xfail(raises=KeyError), id="Empty"),
        pytest.param(
            {"item": "CircularAperture", "radius": 3.14},
            CircularAperture(radius=3.14),
            id="CircularAperture",
        ),
        pytest.param(
            {"item": "ThinLens", "nwaves": 1.1, "radius": 2.2},
            ThinLens(nwaves=1.1, radius=2.2),
            id="ThinLens",
        ),
        pytest.param(
            {"item": "SquareAperture", "size": 1},
            SquareAperture(size=1),
            id="SquareAperture",
        ),
        pytest.param(
            {"item": "RectangularAperture", "width": 1.1, "height": 2.2},
            RectangleAperture(width=1.1, height=2.2),
            id="RectangularAperture",
        ),
        pytest.param(
            {"item": "HexagonAperture", "side": 1.1},
            HexagonAperture(side=1.1),
            id="HexagonAperture",
        ),
        pytest.param(
            {"item": "MultiHexagonalAperture", "side": 1.1, "rings": 2.2, "gap": 3.3},
            MultiHexagonalAperture(side=1.1, rings=2.2, gap=3.3),
            id="MultiHexagonalAperture",
        ),
        pytest.param(
            {
                "item": "SecondaryObscuration",
                "secondary_radius": 1.1,
                "n_supports": 2.2,
                "support_width": 3.3,
            },
            SecondaryObscuration(
                secondary_radius=1.1, n_supports=2.2, support_width=3.3
            ),
            id="SecondaryObscuration",
        ),
        pytest.param(
            {
                "item": "ZernikeWFE",
                "radius": 1.1,
                "coefficients": [2.2, 3.3],
                "aperture_stop": 4.4,
            },
            ZernikeWFE(radius=1.1, coefficients=[2.2, 3.3], aperture_stop=4.4),
            id="ZernikeWFE",
        ),
        pytest.param(
            {
                "item": "SineWaveWFE",
                "spatialfreq": 2.2,
                "amplitude": 3.3,
                "rotation": 4.4,
            },
            SineWaveWFE(spatialfreq=2.2, amplitude=3.3, rotation=4.4),
            id="SineWaveWFE",
        ),
    ],
)
def test_create_optical_parameter(dct: t.Mapping, exp_parameter):
    """Test function 'create_optical_parameter."""
    parameter = create_optical_parameter(dct)

    assert parameter == exp_parameter

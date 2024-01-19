from contextlib import AbstractContextManager
from enum import Enum, auto

import numpy as np
import pytest
import xarray as xr

from pyxel.data_structure import Photon
from pyxel.detectors import Geometry


class Factory(Enum):
    EMPTY = auto()
    PHOTON2D = auto()
    PHOTON3D = auto()

    def build(self) -> Photon:
        if self == Factory.EMPTY:
            return Photon(geo=Geometry(row=3, col=3))

        elif self == Factory.PHOTON2D:
            obj = Photon(geo=Geometry(row=3, col=4))
            obj.array = xr.DataArray(
                np.array(
                    [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                    dtype=float,
                ),
                dims=["wavelength", "y", "x"],
            )

            return obj

        elif self == Factory.PHOTON3D:
            obj = Photon(geo=Geometry(row=3, col=4))
            obj.array_3d = xr.DataArray(
                np.array(
                    [
                        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                        [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
                    ],
                    dtype=float,
                ),
                dims=["wavelength", "y", "x"],
            )

            return obj
        else:
            raise NotImplementedError


@pytest.mark.parametrize(
    "name, exp_value, exp_shape",
    [
        pytest.param(Factory.EMPTY, pytest.raises(ValueError), (), id="empty"),
        pytest.param(
            Factory.PHOTON3D,
            np.array(
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                dtype=float,
            ),
            (3, 4),
            id="2D",
        ),
        pytest.param(
            Factory.PHOTON3D,
            np.array(
                [
                    [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                    [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
                ],
                dtype=float,
            ),
            (2, 3, 4),
            id="3D",
        ),
    ],
)
def test_photon(name: Factory, exp_value, exp_shape):
    """Test methods Photon.__array__, .shape and .ndim."""
    photon = name.build()
    assert isinstance(photon, Photon)

    # Test Photon.__array__
    if isinstance(exp_value, AbstractContextManager):
        with exp_value:
            _ = np.array(photon)
    else:
        value = np.array(photon)
        np.testing.assert_equal(value, exp_value)

    # Test Photon.shape
    assert photon.shape == exp_shape

    # Test Photon.ndim
    assert photon.ndim == len(exp_shape)


@pytest.mark.parametrize("obj1", [Factory.EMPTY, Factory.PHOTON2D, Factory.PHOTON3D])
@pytest.mark.parametrize("obj2", [Factory.EMPTY, Factory.PHOTON2D, Factory.PHOTON3D])
def test_eq(obj1: Factory, obj2: Factory):
    """Test method Photon.__eq__"""
    photon1: Photon = obj1.build()
    photon2: Photon = obj2.build()

    if obj1 == obj2:
        assert photon1 == photon2
    else:
        assert photon1 != photon2


def test_add():
    """Test method Photon.__add__"""
    photon: Photon = Factory.EMPTY.build()
    raise NotImplementedError

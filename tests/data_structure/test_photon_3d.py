#   Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this Pyxel package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.

import numpy as np
import pytest
import xarray as xr

from pyxel.data_structure import Photon3D
from pyxel.detectors import Geometry


@pytest.fixture
def valid_geometry() -> Geometry:
    """Get a valid Geometry."""
    return Geometry(row=3, col=4)


@pytest.fixture
def valid_data_array() -> xr.DataArray:
    return xr.DataArray(
        np.array(
            [
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
            ],
            dtype=float,
        ),
        dims=["wavelength", "y", "x"],
    )


@pytest.fixture
def empty_photon_3d(valid_geometry: Geometry) -> Photon3D:
    """Get an empty 'Photon3D' object."""
    return Photon3D(geo=valid_geometry)


@pytest.fixture
def photon3d_valid(
    valid_geometry: Geometry, valid_data_array: xr.DataArray
) -> Photon3D:
    """Return a valid 'Photon3D' object."""
    obj = Photon3D(geo=valid_geometry)
    obj.array = valid_data_array

    return obj


@pytest.fixture
def other_photon3d_valid(
    valid_geometry: Geometry, valid_data_array: xr.DataArray
) -> Photon3D:
    """Return a valid 'Photon3D' object."""
    obj = Photon3D(geo=valid_geometry)
    obj.array = valid_data_array * 10

    return obj


def test_photon3d(valid_geometry: Geometry):
    """Test class 'Photon3D'."""
    obj = Photon3D(geo=valid_geometry)

    with pytest.raises(ValueError, match="is not initialized"):
        _ = obj.array


@pytest.mark.parametrize("dtype", [float, np.float16, np.float32, np.float64])
def test_photon3d_set_array(photon3d_valid: Photon3D, dtype):
    """Test property 'Photon3D.array' with a valid array."""
    obj = photon3d_valid
    assert isinstance(obj, Photon3D)

    obj.array = xr.DataArray(
        np.array(
            [
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
            ],
            dtype=dtype,
        ),
        dims=["wavelength", "y", "x"],
    )


@pytest.mark.parametrize("dtype", [int, np.int16, np.uint32])
def test_photon3d_set_array_bad_dtype(photon3d_valid: Photon3D, dtype):
    """Test property 'Photon3D.array' with invalid 'dtype'."""
    obj = photon3d_valid
    assert isinstance(obj, Photon3D)

    new_array = xr.DataArray(
        np.array(
            [
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
            ],
            dtype=dtype,
        ),
        dims=["wavelength", "y", "x"],
    )

    with pytest.raises(ValueError, match="Expected valid dtype"):
        obj.array = new_array


def test_photon3d_set_array_bad_ndim(photon3d_valid: Photon3D):
    """Test property 'Photon3D.array' with invalid 'ndim'."""
    obj = photon3d_valid
    assert isinstance(obj, Photon3D)

    data_2d = xr.DataArray(
        np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], dtype=float),
        dims=["y", "x"],
    )

    with pytest.raises(ValueError, match="Expected array with 3 dimensions"):
        obj.array = data_2d


def test_photon3d_set_array_bad_dimensions(photon3d_valid: Photon3D):
    """Test property 'Photon3D.array' with invalid dimensions'."""
    obj = photon3d_valid
    assert isinstance(obj, Photon3D)

    new_array = xr.DataArray(
        np.array(
            [
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
            ],
            dtype=float,
        ),
        dims=["foo", "y", "x"],
    )

    with pytest.raises(ValueError, match="Expected dimensions"):
        obj.array = new_array


def test_photon3d_set_array_bad_shape(photon3d_valid: Photon3D):
    """Test property 'Photon3D.array' with invalid shape'."""
    obj = photon3d_valid
    assert isinstance(obj, Photon3D)

    with pytest.raises(ValueError, match="Expected shape"):
        obj.array = xr.DataArray(
            np.array(
                [
                    [[0.0, 12.0], [4.0, 16.0], [8.0, 20.0]],
                    [[1.0, 13.0], [5.0, 17.0], [9.0, 21.0]],
                    [[2.0, 14.0], [6.0, 18.0], [10.0, 22.0]],
                    [[3.0, 15.0], [7.0, 19.0], [11.0, 23.0]],
                ]
            ),
            dims=["wavelength", "y", "x"],
        )


def test_eq(photon3d_valid: Photon3D):
    """Test method 'Photon3D.__eq__'."""
    assert photon3d_valid == photon3d_valid


def test_eq_empty(empty_photon_3d: Photon3D):
    """Test method 'Photon3d.__eq__' with empty Photon3D objects."""
    assert empty_photon_3d == empty_photon_3d


def test_not_eq(photon3d_valid: Photon3D, other_photon3d_valid: Photon3D):
    """Test method 'Photon3D.__eq__'."""
    assert photon3d_valid != other_photon3d_valid


def test_to_xarray(photon3d_valid: Photon3D, valid_data_array: xr.DataArray):
    """Test method 'Photon3D.to_xarray()."""
    obj = photon3d_valid.to_xarray()
    assert isinstance(obj, xr.DataArray)
    assert obj.dtype == float

    xr.testing.assert_equal(obj, valid_data_array)


def test_to_dict_from_dict(valid_geometry: Geometry, photon3d_valid: Photon3D):
    """Test methods 'Photon3D.to_dict' and 'Photon3D.from_dict'."""
    dct = photon3d_valid.to_dict()
    assert isinstance(dct, dict)

    new_photon3d = Photon3D.from_dict(geometry=valid_geometry, data=dct)
    assert isinstance(new_photon3d, Photon3D)

    assert photon3d_valid == new_photon3d

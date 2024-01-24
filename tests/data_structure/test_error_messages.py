#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import pytest

from pyxel.detectors import CCD, CCDGeometry, Characteristics, Environment


@pytest.fixture
def empty_detector() -> CCD:
    """Create a detector object with empty Data Containers."""
    return CCD(
        geometry=CCDGeometry(row=10, col=10),
        environment=Environment(),
        characteristics=Characteristics(),
    )


def test_error_message_empty_photon(empty_detector: CCD):
    """Test error messages generated with an empty 'Photon' container."""
    assert isinstance(empty_detector, CCD)

    # Test '.photon.array'
    with pytest.raises(
        ValueError,
        match=r"Consider using the \'illumination\' model from the \'Photon Collection\' group",
    ):
        _ = empty_detector.photon.array


def test_error_message_empty_pixel(empty_detector: CCD):
    """Test error messages generated with an empty 'Pixel' container."""
    assert isinstance(empty_detector, CCD)

    # Test '.pixel.array'
    with pytest.raises(
        ValueError,
        match=r"Consider using the \'simple_collection\' model from the \'Charge Collection\' group",
    ):
        _ = empty_detector.pixel.array


def test_error_message_empty_signal(empty_detector: CCD):
    """Test error messages generated with an empty 'Signal' container."""
    assert isinstance(empty_detector, CCD)

    # Test '.pixel.array'
    with pytest.raises(
        ValueError,
        match=r"Consider using the \'simple_measurement\' model from the \'Charge Measurement\' group",
    ):
        _ = empty_detector.signal.array


def test_error_message_empty_image(empty_detector: CCD):
    """Test error messages generated with an empty 'Image' container."""
    assert isinstance(empty_detector, CCD)

    # Test '.pixel.array'
    with pytest.raises(
        ValueError,
        match=r"Consider using the \'simple_amplifier\' model from the \'Readout Electronics\' group",
    ):
        _ = empty_detector.image.array


# def test_error_message_empty_photon_3d(empty_detector: CCD):
#     """Test error messages generated with an empty 'Photon3d' container."""
#     assert isinstance(empty_detector, CCD)
#
#     # Test '.pixel.array'
#     with pytest.raises(
#         ValueError,
#         match=r"Consider using the \'\' model from the \'Readout Electronics\' group",
#     ):
#         _ = empty_detector.photon3d.array

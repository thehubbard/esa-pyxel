#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#

import pytest

from pyxel.detectors import CCD, CCDCharacteristics, CCDGeometry, Environment, Material


@pytest.fixture
def environment() -> Environment:
    return Environment()


@pytest.fixture
def material() -> Material:
    return Material()


@pytest.fixture
def ccd_geometry() -> CCDGeometry:
    return CCDGeometry()


@pytest.fixture
def ccd_characteristics() -> CCDCharacteristics:
    return CCDCharacteristics()


@pytest.fixture
def CCD_empty(ccd_geometry, material, environment, ccd_characteristics) -> CCD:
    detector = CCD(
        geometry=ccd_geometry,
        material=material,
        environment=environment,
        characteristics=ccd_characteristics,
    )
    return detector

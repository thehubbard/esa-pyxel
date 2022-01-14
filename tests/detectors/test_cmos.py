#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from copy import deepcopy

import numpy as np
import pytest

from pyxel.detectors import (
    CMOS,
    CMOSCharacteristics,
    CMOSGeometry,
    Detector,
    Environment,
    Material,
)
from pyxel.detectors.material import MaterialType


@pytest.fixture
def valid_cmos() -> CMOS:
    """Create a valid `CMOS`."""
    return CMOS(
        geometry=CMOSGeometry(
            row=100,
            col=120,
            total_thickness=123.1,
            pixel_horz_size=12.4,
            pixel_vert_size=34.5,
        ),
        material=Material(
            trapped_charge=None,
            n_acceptor=1.0,
            n_donor=2.0,
            material=MaterialType.HXRG,
            material_density=3.0,
            ionization_energy=4.0,
            band_gap=5.0,
            e_effective_mass=1e-12,
        ),
        environment=Environment(
            temperature=100.1,
            total_ionising_dose=200.2,
            total_non_ionising_dose=300.3,
        ),
        characteristics=CMOSCharacteristics(
            quantum_efficiency=0.1,
            charge_to_volt_conversion=0.2,
            pre_amplification=3.3,
            full_well_capacity=4.4,
        ),
    )


@pytest.mark.parametrize(
    "other_obj, is_equal",
    [
        pytest.param(None, False, id="None"),
        pytest.param(
            CMOS(
                geometry=CMOSGeometry(),
                material=Material(),
                environment=Environment(),
                characteristics=CMOSCharacteristics(),
            ),
            False,
            id="Empty 'CMOS'",
        ),
        pytest.param(
            CMOS(
                geometry=CMOSGeometry(
                    row=100,
                    col=120,
                    total_thickness=123.1,
                    pixel_horz_size=12.4,
                    pixel_vert_size=34.5,
                ),
                material=Material(),
                environment=Environment(),
                characteristics=CMOSCharacteristics(),
            ),
            False,
            id="Almost same parameters, different class",
        ),
        pytest.param(
            CMOS(
                geometry=CMOSGeometry(
                    row=100,
                    col=120,
                    total_thickness=123.1,
                    pixel_horz_size=12.4,
                    pixel_vert_size=34.5,
                ),
                material=Material(
                    trapped_charge=None,
                    n_acceptor=1.0,
                    n_donor=2.0,
                    material=MaterialType.HXRG,
                    material_density=3.0,
                    ionization_energy=4.0,
                    band_gap=5.0,
                    e_effective_mass=1e-12,
                ),
                environment=Environment(
                    temperature=100.1,
                    total_ionising_dose=200.2,
                    total_non_ionising_dose=300.3,
                ),
                characteristics=CMOSCharacteristics(
                    quantum_efficiency=0.1,
                    charge_to_volt_conversion=0.2,
                    pre_amplification=3.3,
                    full_well_capacity=4.4,
                ),
            ),
            True,
            id="Same parameters, same class",
        ),
    ],
)
def test_is_equal(valid_cmos: CMOS, other_obj, is_equal):
    """Test equality statement for `CMOS`."""

    if is_equal:
        assert valid_cmos == other_obj
    else:
        assert valid_cmos != other_obj


def comparison(dct, other_dct):
    assert set(dct) == set(other_dct) == {"type", "data", "properties"}
    assert dct["properties"] == other_dct["properties"]
    np.testing.assert_equal(dct["data"], other_dct["data"])


@pytest.mark.parametrize("klass", [CMOS, Detector])
@pytest.mark.parametrize(
    "obj, exp_dict",
    [
        (
            CMOS(
                geometry=CMOSGeometry(
                    row=100,
                    col=120,
                    total_thickness=123.1,
                    pixel_horz_size=12.4,
                    pixel_vert_size=34.5,
                ),
                material=Material(
                    trapped_charge=None,
                    n_acceptor=1.0,
                    n_donor=2.0,
                    material=MaterialType.HXRG,
                    material_density=3.0,
                    ionization_energy=4.0,
                    band_gap=5.0,
                    e_effective_mass=1e-12,
                ),
                environment=Environment(
                    temperature=100.1,
                    total_ionising_dose=200.2,
                    total_non_ionising_dose=300.3,
                ),
                characteristics=CMOSCharacteristics(
                    quantum_efficiency=0.1,
                    charge_to_volt_conversion=0.2,
                    pre_amplification=3.3,
                    full_well_capacity=4.4,
                ),
            ),
            {
                "type": "cmos",
                "properties": {
                    "geometry": {
                        "row": 100,
                        "col": 120,
                        "total_thickness": 123.1,
                        "pixel_horz_size": 12.4,
                        "pixel_vert_size": 34.5,
                    },
                    "material": {
                        "trapped_charge": None,
                        "n_acceptor": 1.0,
                        "n_donor": 2.0,
                        "material": "hxrg",
                        "material_density": 3.0,
                        "ionization_energy": 4.0,
                        "band_gap": 5.0,
                        "e_effective_mass": 1e-12,
                    },
                    "environment": {
                        "temperature": 100.1,
                        "total_ionising_dose": 200.2,
                        "total_non_ionising_dose": 300.3,
                    },
                    "characteristics": {
                        "quantum_efficiency": 0.1,
                        "charge_to_volt_conversion": 0.2,
                        "pre_amplification": 3.3,
                        "full_well_capacity": 4.4,
                    },
                },
                "data": {
                    "photon": np.zeros(shape=(100, 120)),
                    "pixel": np.zeros(shape=(100, 120)),
                    "signal": np.zeros(shape=(100, 120)),
                    "image": np.zeros(shape=(100, 120)),
                    "charge": None,
                },
            },
        )
    ],
)
def test_to_and_from_dict(klass, obj, exp_dict):
    """Test methods 'to_dict', 'from_dict'."""
    assert type(obj) == CMOS

    # Convert from `CMOS` to a `dict`
    dct = obj.to_dict()
    comparison(dct, exp_dict)

    # Copy 'exp_dict'
    copied_dct = deepcopy(exp_dict)  # create a new dict
    assert copied_dct is not exp_dict
    comparison(copied_dct, exp_dict)

    # Convert from `dict` to `CMOS`
    other_obj = klass.from_dict(copied_dct)
    assert type(other_obj) == CMOS
    assert obj == other_obj
    assert obj is not other_obj
    comparison(copied_dct, exp_dict)

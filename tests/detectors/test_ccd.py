#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from copy import deepcopy

import pytest

from pyxel.detectors import (
    CCD,
    CCDCharacteristics,
    CCDGeometry,
    Detector,
    Environment,
    Material,
)
from pyxel.detectors.material import MaterialType


@pytest.fixture
def valid_ccd() -> CCD:
    """Create a valid `CCD`."""
    return CCD(
        geometry=CCDGeometry(
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
        characteristics=CCDCharacteristics(
            qe=0.1,
            eta=0.2,
            sv=3.3,
            amp=4.4,
            a1=5,
            a2=6,
            fwc=7,
            vg=0.8,
            dt=9.9,
            fwc_serial=10,
            svg=0.11,
            t=0.12,
            st=0.13,
        ),
    )


@pytest.mark.parametrize(
    "other_obj, is_equal",
    [
        pytest.param(None, False, id="None"),
        pytest.param(
            CCD(
                geometry=CCDGeometry(),
                material=Material(),
                environment=Environment(),
                characteristics=CCDCharacteristics(),
            ),
            False,
            id="Empty 'CCD'",
        ),
        pytest.param(
            CCD(
                geometry=CCDGeometry(
                    row=100,
                    col=120,
                    total_thickness=123.1,
                    pixel_horz_size=12.4,
                    pixel_vert_size=34.5,
                ),
                material=Material(),
                environment=Environment(),
                characteristics=CCDCharacteristics(),
            ),
            False,
            id="Almost same parameters, different class",
        ),
        pytest.param(
            CCD(
                geometry=CCDGeometry(
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
                characteristics=CCDCharacteristics(
                    qe=0.1,
                    eta=0.2,
                    sv=3.3,
                    amp=4.4,
                    a1=5,
                    a2=6,
                    fwc=7,
                    vg=0.8,
                    dt=9.9,
                    fwc_serial=10,
                    svg=0.11,
                    t=0.12,
                    st=0.13,
                ),
            ),
            True,
            id="Same parameters, same class",
        ),
    ],
)
def test_is_equal(valid_ccd: CCD, other_obj, is_equal):
    """Test equality statement for `CCD`."""

    if is_equal:
        assert valid_ccd == other_obj
    else:
        assert valid_ccd != other_obj


@pytest.mark.parametrize("klass", [CCD, Detector])
@pytest.mark.parametrize(
    "obj, exp_dict",
    [
        (
            CCD(
                geometry=CCDGeometry(
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
                characteristics=CCDCharacteristics(
                    qe=0.1,
                    eta=0.2,
                    sv=3.3,
                    amp=4.4,
                    a1=5,
                    a2=6,
                    fwc=7,
                    vg=0.8,
                    dt=9.9,
                    fwc_serial=10,
                    svg=0.11,
                    t=0.12,
                    st=0.13,
                ),
            ),
            {
                "type": "ccd",
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
                    "qe": 0.1,
                    "eta": 0.2,
                    "sv": 3.3,
                    "amp": 4.4,
                    "a1": 5,
                    "a2": 6,
                    "fwc": 7,
                    "vg": 0.8,
                    "dt": 9.9,
                    "fwc_serial": 10,
                    "svg": 0.11,
                    "t": 0.12,
                    "st": 0.13,
                },
            },
        )
    ],
)
def test_to_and_from_dict(klass, obj, exp_dict):
    """Test methods 'to_dict', 'from_dict'."""
    assert type(obj) == CCD

    # Convert from `CCD` to a `dict`
    dct = obj.to_dict()
    assert dct == exp_dict

    # Copy 'exp_dict'
    copied_dct = deepcopy(exp_dict)  # create a new dict
    assert copied_dct is not exp_dict
    assert copied_dct == exp_dict

    # Convert from `dict` to `CCD`
    other_obj = klass.from_dict(copied_dct)
    assert type(other_obj) == CCD
    assert obj == other_obj
    assert obj is not other_obj
    assert copied_dct == exp_dict

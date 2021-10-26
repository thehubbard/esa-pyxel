#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from copy import deepcopy

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
            n_output=2,
            n_row_overhead=13,
            n_frame_overhead=14,
            reverse_scan_direction=True,
            reference_pixel_border_width=8,
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
            qe=0.1,
            eta=0.2,
            sv=3.3,
            amp=4.4,
            a1=5,
            a2=6,
            fwc=7,
            vg=0.8,
            dt=9.9,
            cutoff=10.0,
            vbiaspower=1.0,
            dsub=0.5,
            vreset=0.2,
            biasgate=2.0,
            preampref=3.0,
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
                    n_output=2,
                    n_row_overhead=13,
                    n_frame_overhead=14,
                    reverse_scan_direction=True,
                    reference_pixel_border_width=8,
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
                    n_output=2,
                    n_row_overhead=13,
                    n_frame_overhead=14,
                    reverse_scan_direction=True,
                    reference_pixel_border_width=8,
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
                    qe=0.1,
                    eta=0.2,
                    sv=3.3,
                    amp=4.4,
                    a1=5,
                    a2=6,
                    fwc=7,
                    vg=0.8,
                    dt=9.9,
                    cutoff=10.0,
                    vbiaspower=1.0,
                    dsub=0.5,
                    vreset=0.2,
                    biasgate=2.0,
                    preampref=3.0,
                ),
            ),
            True,
            id="Almost same parameters, different class",
        ),
    ],
)
def test_is_equal(valid_cmos: CMOS, other_obj, is_equal):
    """Test equality statement for `CMOS`."""

    if is_equal:
        assert valid_cmos == other_obj
    else:
        assert valid_cmos != other_obj


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
                    n_output=2,
                    n_row_overhead=13,
                    n_frame_overhead=14,
                    reverse_scan_direction=True,
                    reference_pixel_border_width=8,
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
                    qe=0.1,
                    eta=0.2,
                    sv=3.3,
                    amp=4.4,
                    a1=5,
                    a2=6,
                    fwc=7,
                    vg=0.8,
                    dt=9.9,
                    cutoff=10.0,
                    vbiaspower=1.0,
                    dsub=0.5,
                    vreset=0.2,
                    biasgate=2.0,
                    preampref=3.0,
                ),
            ),
            {
                "type": "cmos",
                "geometry": {
                    "row": 100,
                    "col": 120,
                    "total_thickness": 123.1,
                    "pixel_horz_size": 12.4,
                    "pixel_vert_size": 34.5,
                    "n_output": 2,
                    "n_row_overhead": 13,
                    "n_frame_overhead": 14,
                    "reverse_scan_direction": True,
                    "reference_pixel_border_width": 8,
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
                    "cutoff": 10.0,
                    "vbiaspower": 1.0,
                    "dsub": 0.5,
                    "vreset": 0.2,
                    "biasgate": 2.0,
                    "preampref": 3.0,
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
    assert dct == exp_dict

    # Copy 'exp_dict'
    copied_dct = deepcopy(exp_dict)  # create a new dict
    assert copied_dct is not exp_dict
    assert copied_dct == exp_dict

    # Convert from `dict` to `CMOS`
    other_obj = klass.from_dict(copied_dct)
    assert type(other_obj) == CMOS
    assert obj == other_obj
    assert obj is not other_obj
    assert copied_dct == exp_dict

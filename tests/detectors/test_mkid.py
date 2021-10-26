#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from copy import deepcopy

import pytest

from pyxel.detectors import (
    MKID,
    Detector,
    Environment,
    Material,
    MKIDCharacteristics,
    MKIDGeometry,
)
from pyxel.detectors.material import MaterialType


@pytest.fixture
def valid_mkid() -> MKID:
    """Create a valid `MKID`."""
    return MKID(
        geometry=MKIDGeometry(
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
        characteristics=MKIDCharacteristics(
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
            MKID(
                geometry=MKIDGeometry(),
                material=Material(),
                environment=Environment(),
                characteristics=MKIDCharacteristics(),
            ),
            False,
            id="Empty 'MKID'",
        ),
        pytest.param(
            MKID(
                geometry=MKIDGeometry(
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
                characteristics=MKIDCharacteristics(),
            ),
            False,
            id="Almost same parameters, different class",
        ),
        pytest.param(
            MKID(
                geometry=MKIDGeometry(
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
                characteristics=MKIDCharacteristics(
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
            id="Same parameters, same class",
        ),
    ],
)
def test_is_equal(valid_mkid: MKID, other_obj, is_equal):
    """Test equality statement for `MKID`."""

    if is_equal:
        assert valid_mkid == other_obj
    else:
        assert valid_mkid != other_obj


@pytest.mark.parametrize("klass", [MKID, Detector])
@pytest.mark.parametrize(
    "obj, exp_dict",
    [
        (
            MKID(
                geometry=MKIDGeometry(
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
                characteristics=MKIDCharacteristics(
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
                "type": "mkid",
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
    assert type(obj) == MKID

    # Convert from `MKID` to a `dict`
    dct = obj.to_dict()
    assert dct == exp_dict

    # Copy 'exp_dict'
    copied_dct = deepcopy(exp_dict)  # create a new dict
    assert copied_dct is not exp_dict
    assert copied_dct == exp_dict

    # Convert from `dict` to `MKID`
    other_obj = klass.from_dict(copied_dct)
    assert type(other_obj) == MKID
    assert obj == other_obj
    assert obj is not other_obj
    assert copied_dct == exp_dict

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
from copy import deepcopy

import pytest

from pyxel.detectors import Material
from pyxel.detectors.material import MaterialType


@pytest.mark.parametrize(
    "trapped_charge, n_acceptor, n_donor, material, material_density, ionization_energy, band_gap, e_effective_mass",
    [
        (None, 0.0, 0.0, MaterialType.Silicon, 0, 0.0, 0, 0),
        (None, 1000.0, 1000.0, MaterialType.HXRG, 10000.0, 100.0, 10.0, 1.0e-10),
    ],
)
def test_create_valid_material(
    trapped_charge,
    n_acceptor,
    n_donor,
    material,
    material_density,
    ionization_energy,
    band_gap,
    e_effective_mass,
):
    """Test when creating a valid `Material` object."""
    _ = Material(
        trapped_charge=trapped_charge,
        n_acceptor=n_acceptor,
        n_donor=n_donor,
        material=material,
        material_density=material_density,
        ionization_energy=ionization_energy,
        band_gap=band_gap,
        e_effective_mass=e_effective_mass,
    )


@pytest.mark.parametrize(
    "trapped_charge, n_acceptor, n_donor, material, material_density, ionization_energy, band_gap, e_effective_mass,exp_exc",
    [
        pytest.param(
            None,
            -0.1,
            20.0,
            MaterialType.Silicon,
            30.0,
            40.0,
            1.2,
            0.0,
            ValueError,
            id="n_acceptor < 0",
        ),
        pytest.param(
            None,
            1000.1,
            20.0,
            MaterialType.Silicon,
            30.0,
            40.0,
            1.2,
            0.0,
            ValueError,
            id="n_acceptor > 1000",
        ),
        pytest.param(
            None,
            10.0,
            -0.1,
            MaterialType.Silicon,
            30.0,
            40.0,
            1.2,
            0.0,
            ValueError,
            id="n_donor < 0",
        ),
        pytest.param(
            None,
            10.0,
            1000.1,
            MaterialType.Silicon,
            30.0,
            40.0,
            1.2,
            0.0,
            ValueError,
            id="n_donor > 1000",
        ),
        pytest.param(
            None,
            10.0,
            20.0,
            MaterialType.Silicon,
            -0.1,
            40.0,
            1.2,
            0.0,
            ValueError,
            id="material_density < 0",
        ),
        pytest.param(
            None,
            10.0,
            20.0,
            MaterialType.Silicon,
            10000.1,
            40.0,
            1.2,
            0.0,
            ValueError,
            id="material_density > 10000",
        ),
        pytest.param(
            None,
            10.0,
            20.0,
            MaterialType.Silicon,
            30.0,
            -0.1,
            1.2,
            0.0,
            ValueError,
            id="ionization_energy < 0",
        ),
        pytest.param(
            None,
            10.0,
            20.0,
            MaterialType.Silicon,
            30.0,
            100.1,
            1.2,
            0.0,
            ValueError,
            id="ionization_energy > 100",
        ),
        pytest.param(
            None,
            10.0,
            20.0,
            MaterialType.Silicon,
            30.0,
            40.0,
            -0.1,
            0.0,
            ValueError,
            id="band_gap < 0",
        ),
        pytest.param(
            None,
            10.0,
            20.0,
            MaterialType.Silicon,
            30.0,
            40.0,
            10.1,
            0.0,
            ValueError,
            id="band_gap > 10",
        ),
        pytest.param(
            None,
            10.0,
            20.0,
            MaterialType.Silicon,
            30.0,
            40.0,
            1.2,
            -0.1,
            ValueError,
            id="e_effective_mass < 0",
        ),
        pytest.param(
            None,
            10.0,
            20.0,
            MaterialType.Silicon,
            30.0,
            40.0,
            1.2,
            1.1e-10,
            ValueError,
            id="e_effective_mass > 1e-10",
        ),
        # pytest.param(None, 10., 20., MaterialType.Silicon, 30., 40., 1.2, 0., ValueError,id='foo'),
    ],
)
def test_create_invalid_material(
    trapped_charge,
    n_acceptor,
    n_donor,
    material,
    material_density,
    ionization_energy,
    band_gap,
    e_effective_mass,
    exp_exc,
):
    """Test when creating an invalid `Material` object."""
    with pytest.raises(exp_exc):
        _ = Material(
            trapped_charge=trapped_charge,
            n_acceptor=n_acceptor,
            n_donor=n_donor,
            material=material,
            material_density=material_density,
            ionization_energy=ionization_energy,
            band_gap=band_gap,
            e_effective_mass=e_effective_mass,
        )


@pytest.mark.parametrize(
    "other_obj, is_equal",
    [
        pytest.param(None, False, id="None"),
        pytest.param(Material(), False, id="Empty 'Material'"),
        pytest.param(Material(n_acceptor=1.0), False, id="Only one parameter"),
        pytest.param(
            Material(
                trapped_charge=None,
                n_acceptor=1.0,
                n_donor=2.0,
                material=MaterialType.HXRG,
                material_density=3.0,
                ionization_energy=4.0,
                band_gap=5.0,
                e_effective_mass=1e-12,
            ),
            True,
            id="Same parameters, same class",
        ),
    ],
)
def test_is_equal(other_obj, is_equal):
    """Test equality statement for CCDGeometry."""
    obj = Material(
        trapped_charge=None,
        n_acceptor=1.0,
        n_donor=2.0,
        material=MaterialType.HXRG,
        material_density=3.0,
        ionization_energy=4.0,
        band_gap=5.0,
        e_effective_mass=1e-12,
    )

    if is_equal:
        assert obj == other_obj
    else:
        assert obj != other_obj


@pytest.mark.parametrize(
    "obj, exp_dict",
    [
        (
            Material(
                trapped_charge=None,
                n_acceptor=1.0,
                n_donor=2.0,
                material=MaterialType.HXRG,
                material_density=3.0,
                ionization_energy=4.0,
                band_gap=5.0,
                e_effective_mass=1e-12,
            ),
            {
                "trapped_charge": None,
                "n_acceptor": 1.0,
                "n_donor": 2.0,
                "material": "hxrg",
                "material_density": 3.0,
                "ionization_energy": 4.0,
                "band_gap": 5.0,
                "e_effective_mass": 1e-12,
            },
        )
    ],
)
def test_to_and_from_dict(obj, exp_dict):
    """Test methods 'to_dict', 'from_dict'."""
    assert type(obj) == Material

    # Convert from `Material` to a `dict`
    dct = obj.to_dict()
    assert dct == exp_dict

    # Copy 'exp_dict'
    copied_dct = deepcopy(exp_dict)  # create a new dict
    assert copied_dct is not exp_dict
    assert copied_dct == exp_dict

    # Convert from `dict` to `Material`
    other_obj = Material.from_dict(copied_dct)
    assert type(other_obj) == Material
    assert obj == other_obj
    assert obj is not other_obj
    assert copied_dct == exp_dict

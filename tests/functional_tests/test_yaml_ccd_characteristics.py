"""Test `PyxelLoader` with `CCDChararacteristics`."""

import pytest
from pyxel.detectors.ccd_characteristics import CCDCharacteristics
from pyxel.pipelines.yaml_processor import load


def test_loader():
    """Test `PyxelLoader`."""
    data = """
!ccd_characteristics
    k: 1
    j: 2
    qe: 3
    eta: 4
    sv: 5
    accd: 6
    a1: 7
    a2: 8    
"""

    obj = load(data)

    assert isinstance(obj, CCDCharacteristics)
    assert obj.k == 1
    assert obj.j == 2
    assert obj.qe == 3
    assert obj.eta == 4
    assert obj.sv == 5
    assert obj.accd == 6
    assert obj.a1 == 7
    assert obj.a2 == 8



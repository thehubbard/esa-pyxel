import pytest

from pyxel.detectors import (
    APD,
    CCD,
    APDCharacteristics,
    APDGeometry,
    CCDCharacteristics,
    CCDGeometry,
    Environment,
    ReadoutProperties,
)
from pyxel.models.charge_generation import (
    dark_current,
    dark_current_saphira,
    simple_dark_current,
)
from pyxel.models.charge_transfer.EMCCD_poisson import multiplication_register


@pytest.fixture
def ccd_10x10() -> CCD:
    """Create a valid CCD detector."""
    detector = CCD(
        geometry=CCDGeometry(
            row=10,
            col=10,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(temperature=200.0),
        characteristics=CCDCharacteristics(),
    )
    detector._readout_properties = ReadoutProperties(num_steps=1)
    return detector


def test_multiplication_register(ccd_10x10: CCD):
    detector = ccd_10x10

    multiplication_register(detector=detector, total_gain=0.0, gain_elements=1)


@pytest.mark.parametrize(
    "total_gain,gain_elements",
    [
        (-1, 10),
        (1, -1),
        (-1, -1),
    ],
)
def test_multiplication_register_bad_inputs(ccd_10x10: CCD, total_gain, gain_elements):
    detector = ccd_10x10

    with pytest.raises(ValueError, match="Wrong input parameter"):
        multiplication_register(
            detector=detector, total_gain=total_gain, gain_elements=gain_elements
        )

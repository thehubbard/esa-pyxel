"""TBW."""
from pyxel.detectors.characteristics import Characteristics
import esapy_config.config as ec
from esapy_config import validators


@ec.config(mode='RO')
class CMOSCharacteristics(Characteristics):
    """Characteristical attributes of a CMOS-based detector."""

    cutoff = ec.setting(
        type=float,
        default=2.5,
        validator=validators.validate_range(1.7, 15.),
        doc='Cutoff wavelength',
        metadata={'units': 'um'}
    )

    vbiaspower = ec.setting(
        type=float,
        default=3.350,
        validator=validators.validate_range(0.0, 3.4),
        doc='VBIASPOWER',
        metadata={'units': 'V'}
    )

    dsub = ec.setting(
        type=float,
        default=0.500,
        validator=validators.validate_range(0.3, 1.0),
        doc='DSUB',
        metadata={'units': 'V'}
    )

    vreset = ec.setting(
        type=float,
        default=0.250,
        validator=validators.validate_range(0.0, 0.3),
        doc='VRESET',
        metadata={'units': 'V'}
    )

    biasgate = ec.setting(
        type=float,
        default=2.300,
        validator=validators.validate_range(1.8, 2.6),
        doc='BIASGATE',
        metadata={'units': 'V'}
    )

    preampref = ec.setting(
        type=float,
        default=1.700,
        validator=validators.validate_range(0.0, 4.0),
        doc='PREAMPREF',
        metadata={'units': 'V'}
    )

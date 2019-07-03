"""TBW."""
from pyxel.detectors.characteristics import Characteristics
import esapy_config.config as ec
from esapy_config import validators


@ec.config(mode='RO')
class CCDCharacteristics(Characteristics):
    """Characteristical attributes of a CCD detector."""

    fwc_serial = ec.setting(
        type=int,
        default=0,
        validator=validators.interval(0.0, 1.e+7),
        doc='Full well capacity (serial register)',
        metadata={'units': 'electrons'}                     # FRED: We could create a new `attribute`
                                                            #       specific for Pyxel that will include 'units'
    )
    svg = ec.setting(
        type=float,
        default=0.0,
        validator=validators.interval(0.0, 1.0),
        doc='Half pixel volume charge can occupy (serial register)',  # TODO should be the full volume and not the half
        metadata={'units': 'cm^2'}
    )
    t = ec.setting(
        type=float,
        default=0.0,
        validator=validators.interval(0.0, 10.0),
        doc='Parallel transfer period',
        metadata={'units': 's'}
    )
    st = ec.setting(
        type=float,
        default=0.0,
        validator=validators.interval(0.0, 10.0),
        doc='Serial transfer period',
        metadata={'units': 's'}
    )

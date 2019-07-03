"""TBW."""
import esapy_config.config as ec
from esapy_config import validators


@ec.config(mode='RO')
class Characteristics:
    """Characteristical attributes of the detector."""

    qe = ec.setting(
        type=float,
        default=0.,
        validator=validators.interval(0., 1.),
        doc='Quantum efficiency',
        metadata={'units': ''}
    )
    eta = ec.setting(
        type=float,
        default=0.,
        validator=validators.interval(0., 1.),
        doc='Quantum yield',
        metadata={'units': 'e-/photon'}
    )
    sv = ec.setting(
        type=float,
        default=0.,
        validator=validators.interval(0., 100.),
        doc='Sensitivity of charge readout',
        metadata={'units': 'V/e-'}
    )
    amp = ec.setting(
        type=float,
        default=0.,
        validator=validators.interval(0., 100.),
        doc='Gain of output amplifier',
        metadata={'units': 'V/V'}
    )
    a1 = ec.setting(
        type=float,
        default=0.,
        validator=validators.interval(0., 100.),
        doc='Gain of the signal processor',
        metadata={'units': 'V/V'}
    )
    a2 = ec.setting(
        type=int,
        default=0,
        validator=validators.interval(0, 65536),
        doc='Gain of the Analog-Digital Converter',
        metadata={'units': 'ADU/V'}
    )
    fwc = ec.setting(
        type=int,
        default=0,
        validator=validators.interval(0., 1.e+7),
        doc='Full well capacity',
        metadata={'units': 'e-'}
    )
    vg = ec.setting(
        type=float,
        default=0.,
        validator=validators.interval(0., 1.),
        doc='Half pixel volume charge can occupy',      # TODO should be the full volume and not the half
        metadata={'units': 'cm^2'}
    )
    dt = ec.setting(
        type=float,
        default=0.0,
        validator=validators.interval(0., 10.),
        doc='Pixel dwell time',
        metadata={'units': 's'}
    )

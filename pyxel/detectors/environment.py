"""TBW."""
import esapy_config.config as ec
from esapy_config import validators


@ec.config(mode='RO')
class Environment:
    """Environmental attributes of the detector."""

    temperature = ec.setting(
        type=float,
        default=273.15,
        validator=validators.validate_range(0., 1000.),
        metadata={'units': 'K'},
        doc='Temperature of the detector'
    )
    total_ionising_dose = ec.setting(
        type=float,
        default=0.0,
        validator=validators.validate_range(0., 1.e15),
        metadata={'units': 'MeV/g'},
        doc='Total Ionising Dose (TID) of the detector'
    )
    total_non_ionising_dose = ec.setting(
        type=float,
        default=0.0,
        validator=validators.validate_range(0., 1.e15),
        metadata={'units': 'MeV/g'},
        doc='Total Non-Ionising Dose (TNID) of the detector'
    )

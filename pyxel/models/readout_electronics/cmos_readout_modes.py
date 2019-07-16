#   --------------------------------------------------------------------------
#   Copyright 2019 SCI-FIV, ESA (European Space Agency)
#   --------------------------------------------------------------------------
"""Non-Destructive Readout modes for CMOS-based detectors."""
import logging
from pyxel.detectors.cmos import CMOS
from esapy_config.checkers import check_choices, check_type_function
from esapy_config import funcargs


# FRED: Remove the following decorators
@funcargs.validate
@funcargs.argument(name='mode', label='', units='', validate=check_choices(['uncorrelated', 'CDS', 'Fowler-N', 'UTR']))
@funcargs.argument(name='fowler_samples', label='', units='', validate=check_type_function(int))
@funcargs.argument(name='detector', label='', units='',
                   validate=check_type_function(CMOS))  # TODO this should be automatic
def non_destructive_readout(detector: CMOS,
                            mode: str,
                            fowler_samples: int = 1) -> None:
    """Non-Destructive Readout modes for CMOS-based detectors.

    :param detector: Pyxel Detector object
    :param mode:
    :param fowler_samples:
    """
    logging.info('')
    if not detector.is_non_destructive_readout or not detector.is_dynamic:
        raise ValueError()

    detector.read_out = False
    if mode == 'uncorrelated':
        if detector.time == (detector.end_time - detector.time_step):
            detector.read_out = True
    elif mode == 'CDS':
        if detector.time == detector.time_step or detector.time == (detector.end_time - detector.time_step):
            detector.read_out = True
    elif mode == 'Fowler-N':
        nt = fowler_samples * detector.time_step
        detector.read_out = True
        if nt < detector.time <= detector.end_time - nt:
            detector.read_out = False
    elif mode == 'UTR':
        detector.read_out = True

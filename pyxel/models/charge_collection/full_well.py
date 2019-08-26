
"""Pyxel full well models."""
import logging
from pyxel.detectors.detector import Detector


# @pyxel.validate
# @pyxel.argument(name='', label='', units='', validate=)
def simple_full_well(detector: Detector) -> None:
    """Limiting the amount of charge in pixel due to full well capacity."""
    logging.info('')
    fwc = detector.characteristics.fwc
    if fwc is None:
        raise ValueError('Full Well Capacity is not defined')
    charge_array = detector.pixel.array
    charge_array[charge_array > fwc] = fwc
    detector.pixel.array = charge_array

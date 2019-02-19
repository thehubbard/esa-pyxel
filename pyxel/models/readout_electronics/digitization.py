#   --------------------------------------------------------------------------
#   Copyright 2018 SCI-FIV, ESA (European Space Agency)
#   --------------------------------------------------------------------------
"""Readout electronics model."""
import logging
from pydoc import locate
import numpy as np
import pyxel
from pyxel.detectors.detector import Detector
# from astropy import units as u


@pyxel.validate
@pyxel.argument(name='data_type', label='type of output data array', units='ADU',
                validate=pyxel.check_choices(['numpy.uint16', 'numpy.uint32', 'numpy.uint64',
                                              'numpy.int32', 'numpy.int64']))
def simple_digitization(detector: Detector,
                        data_type: str = 'numpy.uint16'):
    """Digitize signal array mimicking readout electronics.

    :param detector: Pyxel Detector object
    :param data_type: numpy integer type: ``numpy.uint16``, ``numpy.uint32``, ``numpy.uint64``,
                                          ``numpy.int32``, ``numpy.int64``
    """
    logger = logging.getLogger('pyxel')
    logger.info('')

    data_type = locate(data_type)
    if data_type is None:
        raise TypeError('Can not locate the type defined as `data_type` argument in yaml file.')
    # Gain of the Analog-Digital Converter
    detector.signal.array *= detector.characteristics.a2
    # floor of signal values element-wise (quantization)
    detector.signal.array = np.floor(detector.signal.array)
    # convert floats to other datatype (e.g. 16-bit unsigned integers)
    detector.signal.array = np.clip(detector.signal.array, a_min=np.iinfo(data_type).min, a_max=np.iinfo(data_type).max)
    detector.image.array = detector.signal.array.astype(data_type)


def simple_processing(detector: Detector):
    """Create an image array from signal array.

    :param detector: Pyxel Detector object
    """
    logger = logging.getLogger('pyxel')
    logger.info('')
    detector.signal.array *= detector.characteristics.a2
    detector.image.array = detector.signal.array

#   --------------------------------------------------------------------------
#   Copyright 2018 SCI-FIV, ESA (European Space Agency)
#   --------------------------------------------------------------------------
"""Readout electronics model."""
import logging
import numpy as np
from pydoc import locate
import pyxel
from pyxel.detectors.detector import Detector
# from astropy import units as u


@pyxel.validate
@pyxel.argument(name='data_type', label='type of output data array', units='ADU',
                validate=pyxel.check_choices(['numpy.uint16', 'numpy.uint32', 'numpy.uint64',
                                              'numpy.int32', 'numpy.int64', None]))
def simple_digitization(detector: Detector,
                        data_type: str = 'numpy.uint16') -> Detector:
    """Create an image array from signal array mimicking readout electronics.

    :param detector:
    :param data_type: numpy integer type (numpy.uint16, numpy.uint32, numpy.uint64, numpy.int32, numpy.int64)
    :return:
    """
    logging.info('')

    data_type = locate(data_type)
    if data_type is None:
        raise TypeError('Can not locate the type defined as `data_type` argument in yaml file.')

    # floor of signal values element-wise (quantization)
    array = np.floor(detector.signal.array)

    # convert floats to other datatype (e.g. 16-bit unsigned integers)
    np.clip(array, a_min=np.iinfo(data_type).min, a_max=np.iinfo(data_type).max)
    detector.image.array = array.astype(data_type)

    return detector

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Simple digitization."""


import numpy as np

from pyxel.detectors import MKID


# TODO: Fix this
def simple_digitization(detector: MKID, data_type: str = "uint16") -> None:
    """Digitize signal array mimicking readout electronics.

    Parameters
    ----------
    detector : Detector
        Pyxel Detector object.
    data_type : str
        The desired data-type for the array. The data-type must be an signed or
        unsigned integer.
        Examples: ``numpy.uint16``, ``numpy.uint32``, ``numpy.uint64``,
        ``numpy.int32``, ``numpy.int64``
    """
    try:
        d_type = np.dtype(data_type)  # type: np.dtype
    except TypeError as ex:
        raise TypeError(
            "Can not locate the type defined as `data_type` argument in yaml file."
        ) from ex

    # Gain of the Analog-Digital Converter
    detector.signal.array *= detector.characteristics.a2

    # floor of signal values element-wise (quantization)
    detector.signal.array = np.floor(detector.signal.array)

    # convert floats to other datatype (e.g. 16-bit unsigned integers)
    result = np.asarray(
        np.clip(
            detector.signal.array,
            a_min=np.iinfo(d_type).min,
            a_max=np.iinfo(d_type).max,
        )
    )  # type: np.ndarray

    detector.signal.array = result
    detector.image.array = np.asarray(detector.signal.array, dtype=d_type)

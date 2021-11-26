#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Pyxel charge injection functions for CCDs."""
import logging
import typing as t

import numpy as np

from pyxel.detectors import CCD

# TODO: fix docstring, put documentation, private function


# TODO: Fix this
# @validators.validate
# @config.argument(name='detector', label='', units='', validate=checkers.check_type(CCD))
def charge_blocks(
    detector: CCD,
    charge_level: int,
    block_start: int = 0,
    block_end: t.Optional[int] = None,
) -> None:
    """TBW.

    :param detector:
    :param charge_level:
    :param block_start:
    :param block_end:
    :return:
    """
    logging.info("")
    geo = detector.geometry
    if block_end is None:
        block_end = geo.row

    # all pixels has zero charge by default
    detector_charge = np.zeros((geo.row, geo.col))
    detector_charge[slice(block_start, block_end), :] = charge_level

    detector.charge.add_charge_array(detector_charge)

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import logging

from pyxel.detectors import Detector


def dead_time_filter(detector: Detector, dead_time: float) -> None:
    """TBW."""
    logging.info("")

    maximum_count = 1.0 / dead_time

    detector.phase.array[detector.phase.array >= maximum_count] = maximum_count

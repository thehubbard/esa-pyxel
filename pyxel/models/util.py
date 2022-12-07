#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from pathlib import Path
from typing import Union

from pyxel.detectors import Detector


def load_detector(detector: Detector, filename: Union[str, Path]) -> None:
    """Load a new detector from a file."""
    detector = Detector.load(filename)


def save_detector(detector: Detector, filename: Union[str, Path]) -> None:
    """Save the current detector into a file."""
    detector.save(filename)

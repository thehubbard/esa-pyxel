#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
"""Functions for assigning custom QE maps."""

import typing as t
from pyxel.inputs_outputs import load_image
from pathlib import Path

if t.TYPE_CHECKING:
    from pyxel.detectors import Detector


def qe_map(detector: "Detector", filename: t.Union[str, Path]) -> None:
    """Upload from file and assign a custom QE map to a detector."""
    qe = load_image(filename)
    detector.characteristics.qe = qe

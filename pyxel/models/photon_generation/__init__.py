#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""

import warnings

warnings.warn(
    "Group 'photon_generation' is deprecated "
    "and will be removed in version 2.0. "
    "Use group 'photon_collection' instead.",
    DeprecationWarning,
)

# flake8: noqa
from .illumination import illumination
from .load_image import load_image
from .shot_noise import shot_noise
from .stripe_pattern import stripe_pattern

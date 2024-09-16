#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.


from enum import Enum


class ParameterMode(Enum):
    """Parameter mode class."""

    Product = "product"
    Sequential = "sequential"
    Custom = "custom"

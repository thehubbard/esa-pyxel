#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.


from collections.abc import MutableMapping, Sequence
from enum import Enum
from numbers import Number
from typing import Union

import numpy as np

ParametersType = MutableMapping[
    str,
    Union[
        str,
        Number,
        np.ndarray,
        Sequence[Union[str, Number, np.ndarray]],
    ],
]


class ParameterMode(Enum):
    """Parameter mode class."""

    Product = "product"
    Sequential = "sequential"
    Custom = "custom"


def short(s: str) -> str:
    """Split string with . and return the last element."""
    out = s.split(".")[-1]
    return out


def _get_short_name_with_model(name: str) -> str:
    _, _, model_name, _, param_name = name.split(".")

    return f"{model_name}.{param_name}"

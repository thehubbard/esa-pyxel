#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.


from collections.abc import MutableMapping, Sequence
from copy import deepcopy
from enum import Enum
from numbers import Number
from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from pyxel.pipelines import Processor

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


def create_new_processor(
    processor: "Processor",
    parameter_dict: ParametersType,
) -> "Processor":
    """Create a copy of processor and set new attributes from a dictionary before returning it.

    Parameters
    ----------
    processor: Processor
    parameter_dict: dict

    Returns
    -------
    Processor
    """

    new_processor = deepcopy(processor)

    for key in parameter_dict:
        new_processor.set(key=key, value=parameter_dict[key])

    return new_processor

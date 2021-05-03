#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""
import typing as t
from collections import abc
from numbers import Number
from enum import Enum

from typing_extensions import Literal

from pyxel.evaluator import eval_range


class ParameterType(Enum):
    """TBW."""

    # one-dimensional, can be a dataset coordinate
    Simple = "simple"
    # multi-dimensional or loaded from a file in parallel mode, cannot be a dataset coordinate
    Multi = "multi"


class ParameterValues:
    """TBW."""

    def __init__(
        self,
        key: str,
        values: t.Union[
            Literal["_"], t.Sequence[Literal["_"]], t.Sequence[Number], t.Sequence[str]
        ],
        boundaries: t.Optional[t.Tuple[float, float]] = None,
        enabled: bool = True,
        logarithmic: bool = False,
    ):
        # TODO: should these values be checked ?
        assert values == "_" or (
            isinstance(values, abc.Sequence)
            and (
                all([el == "_" for el in values])
                or all([isinstance(el, str) for el in values])
                or all([isinstance(el, Number) for el in values])
            )
        )

        # unique identifier to the step. example: 'detector.geometry.row'
        self._key = key  # type: str
        self._values = values # type: t.Union[Literal["_"], t.Sequence[Literal["_"]], t.Sequence[Number], t.Sequence[str]]

        # short  name identifier: 'row'
        self._short_name = key.split(".")[-1]

        self._enabled = enabled  # type: bool
        self._logarithmic = logarithmic  # type: bool
        self._boundaries = boundaries  # type: t.Optional[t.Tuple[float, float]]

        self._current = None  # type: t.Optional[t.Union[Literal['_'], Number, str]]

        # with eval_range(self._values) as value_list:
        #     if any([x == "_" or isinstance(x, abc.Sequence) for x in value_list]):
        #         self._type = ParameterType("multi")
        #     elif all([x != "_" and isinstance(x, (str, Number)) for x in value_list]):
        #         self._type = ParameterType("simple")
        #     else:
        #         raise ValueError


    def __repr__(self) -> str:
        cls_name = self.__class__.__name__  # type: str

        return f"{cls_name}<key={self.key!r}, values={self.values!r}, enabled={self.enabled!r}>"

    def __len__(self) -> int:
        values = eval_range(self.values)  # type: t.Sequence
        return len(values)

    # TODO: Is method '__contains__' needed ? If yes then this class will act as a `Collections.abc.Sequence`
    def __iter__(self) -> t.Iterator[Number]:
        values = eval_range(self.values)
        for value in values:
            yield value

    @property
    def key(self) -> str:
        """TBW."""
        return self._key

    @property
    def short_name(self) -> str:
        """TBW."""
        return self._short_name

    @property
    def values(
        self,
    ) -> t.Union[
        Literal["_"], t.Sequence[Literal["_"]], t.Sequence[Number], t.Sequence[str]
    ]:
        """TBW."""
        return self._values

    @property
    def enabled(self) -> bool:
        """TBW."""
        return self._enabled

    @property
    def current(self) -> t.Optional[t.Union[Literal["_"], Number, str]]:
        """TBW."""
        return self._current

    @current.setter
    def current(self, value: t.Union[Literal["_"], str, Number]) -> None:
        """TBW."""
        self._current = value

    @property
    def logarithmic(self) -> bool:
        """TBW."""
        return self._logarithmic

    @property
    def boundaries(self) -> t.Optional[t.Tuple[float, float]]:
        """TBW."""
        return self._boundaries

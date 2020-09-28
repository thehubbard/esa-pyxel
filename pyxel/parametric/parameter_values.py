#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""
import collections
import typing as t
from numbers import Number

from typing_extensions import Literal

from pyxel.evaluator import eval_range


class ParameterValues:
    """TBW."""

    def __init__(
        self,
        key: str,
        values: t.Union[
            Literal["_"],
            t.Sequence[Literal["_"]],
            t.Sequence[str],
            t.Sequence[t.Union[Number]],
        ],
        boundaries: t.Optional[t.Tuple[float, float]] = None,
        enabled: bool = True,
        logarithmic: bool = False,
    ):
        # TODO: should these values be checked ?
        assert values == "_" or (
            isinstance(values, collections.Sequence)
            and (
                all([el == "_" for el in values])
                or all([isinstance(el, str) for el in values])
                or all([isinstance(el, Number) for el in values])
            )
        )

        # unique identifier to the step. example: 'detector.geometry.row'
        self._key = key  # type: str
        self._values = (
            values
        )  # type: t.Union[Literal['_'], t.Sequence[Literal['_']], t.Sequence[str], t.Sequence[t.Union[Number]]]

        self._enabled = enabled  # type: bool
        self._logarithmic = logarithmic  # type: bool
        self._boundaries = boundaries  # type: t.Optional[t.Tuple[float, float]]

        self._current = None  # type: t.Optional[t.Union[Literal['_'], str, Number]]

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
    def values(
        self,
    ) -> t.Union[
        Literal["_"],
        t.Sequence[Literal["_"]],
        t.Sequence[str],
        t.Sequence[t.Union[Number]],
    ]:
        """TBW."""
        return self._values

    @property
    def enabled(self) -> bool:
        """TBW."""
        return self._enabled

    @property
    def current(self) -> t.Optional[t.Union[Literal["_"], str, Number]]:
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

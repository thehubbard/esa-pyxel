#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""
import typing as t

from pyxel.evaluator import eval_range


class ParameterValues:
    """TBW."""

    def __init__(
        self,
        key: str,
        values: t.List[t.Union[float, int, str]],
        boundaries: t.Optional[t.Tuple[float, float]] = None,
        enabled: bool = True,
        current=None,
        logarithmic: bool = False,
    ):
        """TBW.

        :param key:
        :param values:
        :param boundaries:
        :param enabled:
        :param current:
        :param logarithmic:
        """
        # TODO: should these values be evaluated?
        self.key = key  # unique identifier to the step. example: detector.geometry.row
        self.values = values  # type: t.List[t.Union[float, int, str]]
        self.enabled = enabled  # type: bool
        self.current = current
        self.logarithmic = logarithmic  # type: bool
        self.boundaries = boundaries  # type: t.Optional[t.Tuple[float, float]]

    def __repr__(self) -> str:
        """TBW."""
        cls_name = self.__class__.__name__  # type: str

        return f"{cls_name}<key={self.key!r}, values={self.values!r}, enabled={self.enabled!r}>"

    def __len__(self) -> int:
        """TBW."""
        values = eval_range(self.values)  # type: list
        return len(values)

    # TODO: Is method '__contains__' needed ? If yes then this class will act as a `Collections.abc.Sequence`
    def __iter__(self) -> t.Iterator[t.Union[float, int]]:
        """TBW."""
        values = eval_range(self.values)
        for value in values:
            yield value

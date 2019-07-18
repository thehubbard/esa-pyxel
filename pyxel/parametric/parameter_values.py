"""TBW."""
import typing as t
from esapy_config import eval_range


class ParameterValues:
    """TBW."""

    def __init__(self,
                 key: str,
                 values: t.List[t.Union[float, int]],
                 boundaries: t.Optional[t.Tuple[float, float]] = None,
                 enabled: bool = True,
                 current=None,
                 logarithmic: bool = False):
        """TBW.

        :param key:
        :param values:
        :param boundaries:
        :param enabled:
        :param current:
        :param logarithmic:
        """
        # TODO: should the values be evaluated?
        self.key = key                              # unique identifier to the step. example: detector.geometry.row
        self.values = values                        # type: t.List[t.Union[float, int]]
        self.enabled = enabled                      # type: bool
        self.current = current
        self.logarithmic = logarithmic              # type: bool
        self.boundaries = boundaries                # type: t.Optional[t.Tuple[float, float]]

    def __len__(self) -> int:
        """TBW."""
        values = eval_range(self.values)
        return len(values)

    # FRED: Do you need magic method '__contains__' ? If yes then this class will act as a `Collections.abc.Sequence`
    def __iter__(self) -> t.Iterator[t.Union[float, int]]:
        """TBW."""
        values = eval_range(self.values)
        for value in values:
            yield value

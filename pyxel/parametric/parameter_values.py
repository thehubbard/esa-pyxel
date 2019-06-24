"""TBW."""
import typing as t
from esapy_config import eval_range


# FRED: Use more typing information
class ParameterValues:
    """TBW."""

    def __init__(self,
                 key: str,
                 values,
                 enabled: bool = True,
                 current=None,
                 logarithmic: bool = False,
                 boundaries: list = None):
        """TBW.

        :param key:
        :param values:
        :param enabled:
        :param current:
        :param logarithmic:
        :param boundaries:
        """
        # TODO: should the values be evaluated?
        self.key = key                              # unique identifier to the step. example: detector.geometry.row
        self.values = values                        # type: t.List[t.Union[float, int]]
        self.enabled = enabled                      # type: bool
        self.current = current
        self.logarithmic = logarithmic              # type: bool
        self.boundaries = boundaries                # type: t.Optional[list]

    # FRED: Is it needed ? all values in this `dict` should me copied.
    def __getstate__(self):
        """TBW."""
        return {
            'key': self.key,
            'values': self.values,
            'enabled': self.enabled,
            'current': self.current,
            'logarithmic': self.logarithmic,
            'boundaries': self.boundaries,
        }

    def __len__(self):
        """TBW."""
        values = eval_range(self.values)
        return len(values)

    # FRED: Do you need magic method '__contains__' ? If yes then this class will act as a `Collections.abc.Sequence`
    def __iter__(self):
        """TBW."""
        values = eval_range(self.values)
        for value in values:
            yield value

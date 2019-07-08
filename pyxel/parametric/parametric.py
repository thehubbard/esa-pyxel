"""TBW."""
import itertools
import typing as t
from copy import deepcopy
import numpy as np
from esapy_config import get_obj_att, get_value
from pyxel.parametric.parameter_values import ParameterValues

if t.TYPE_CHECKING:
    from ..pipelines.processor import Processor
    from ..calibration.calibration import Calibration
    from ..util import Outputs


# FRED: Create a `Enum` for 'parametric_mode' ?
class ParametricAnalysis:
    """TBW."""

    def __init__(self,
                 parametric_mode: str,
                 parameters: t.List[ParameterValues],
                 from_file: t.Optional[str] = None,
                 # HANS: column_range should be of type: t.Tuple[int, int]
                 column_range: t.Optional[t.List[int]] = None):
        """TBW."""
        self.parametric_mode = parametric_mode
        self._parameters = parameters
        self.file = from_file
        self.data = None  # type: t.Optional[np.ndarrat]
        if column_range:  # HANS: ensure that the size == 2
            self.columns = slice(column_range[0], column_range[1])

    # def __getstate__(self):
    #     """TBW."""
    #     return {'parametric_mode': self.parametric_mode}

    @property
    def enabled_steps(self) -> t.List[ParameterValues]:
        """TBW."""
        return [step for step in self._parameters if step.enabled]

    def _parallel(self, processor: "Processor") -> "t.Iterator[Processor]":
        """TBW.

        :param processor:
        :return:
        """
        self.data = np.loadtxt(self.file)[:, self.columns]
        for data_array in self.data:
            i = 0
            new_proc = deepcopy(processor)  # type: Processor
            for step in self.enabled_steps:
                key = step.key
                # HANS: this is confusing code. Explain.
                if step.values == '_':
                    value = data_array[i]
                    i += 1
                elif isinstance(step.values, list) and all(x == '_' for x in step.values[:]):
                    value = data_array[i: i + len(step.values)]
                    i += len(value)
                else:
                    raise ValueError('Only "_" characters (or a list of them) should be used to '
                                     'indicate parameters updated from file in parallel')
                new_proc.set(key, value)
            yield new_proc

    def _sequential(self, processor: "Processor") -> "t.Iterator[Processor]":
        """TBW.

        :param processor:
        :return:
        """
        for step in self.enabled_steps:
            key = step.key
            for value in step:
                step.current = value
                new_proc = deepcopy(processor)  # type: Processor
                new_proc.set(key, value)
                yield new_proc

    def _embedded(self, processor: "Processor") -> "t.Iterator[Processor]":
        """TBW.

        :param processor:
        :return:
        """
        all_steps = self.enabled_steps
        keys = [step.key for step in self.enabled_steps]
        for params in itertools.product(*all_steps):
            new_proc = deepcopy(processor)  # type: Processor
            for key, value in zip(keys, params):
                for step in all_steps:
                    if step.key == key:
                        step.current = value
                new_proc.set(key=key, value=value)
            yield new_proc

    def collect(self, processor: "Processor") -> "t.List[Processor]":
        """TBW."""
        for step in self.enabled_steps:

            # HANS: the string literal expressions are difficult to maintain.
            #     Example: 'pipeline.', '.arguments', '.enabled'
            #     We may want to consider an API for this.
            if 'pipeline.' in step.key:
                model_name = step.key[:step.key.find('.arguments')]
                model_enabled = model_name + '.enabled'  # type: str
                if not processor.get(model_enabled):
                    raise ValueError('The "%s" model referenced in parametric configuration '
                                     'has not been enabled in yaml config!' % model_name)

            if any(x == '_' for x in step.values[:]) and self.parametric_mode != 'parallel':
                raise ValueError('Either define "parallel" as parametric mode or '
                                 'do not use "_" character in "values" field')

        if self.parametric_mode == 'embedded':
            configs = self._embedded(processor)
        elif self.parametric_mode == 'sequential':
            configs = self._sequential(processor)
        elif self.parametric_mode == 'parallel':
            configs = self._parallel(processor)
        else:
            configs = []

        return configs

    def debug(self, processor: "Processor") -> list:
        """TBW."""
        result = []
        configs = self.collect(processor)
        for i, config in enumerate(configs):
            values = []
            for step in self.enabled_steps:
                _, att = get_obj_att(config, step.key)
                value = get_value(config, step.key)
                values.append((att, value))
            print('%d: %r' % (i, values))  # FRED: Use `logging.debug`
            result.append((i, values))
        return result


# FRED: Create a `Enum` for 'mode' ?
class Configuration:
    """TBW."""

    def __init__(self, mode: str,
                 outputs: "t.Optional[Outputs]" = None,
                 parametric: t.Optional[ParametricAnalysis] = None,
                 calibration: "t.Optional[Calibration]" = None,
                 dynamic: t.Optional[bool] = None
                 ):
        """TBW.

        :param mode:
        :param parametric:
        :param calibration:
        """
        if mode in ['single', 'parametric', 'calibration', 'dynamic']:
            self.mode = mode
        else:
            raise ValueError('Non-existing running mode defined for Pyxel in yaml config file.')
        self.outputs = outputs
        self.parametric = parametric
        self.calibration = calibration
        self.dynamic = dynamic

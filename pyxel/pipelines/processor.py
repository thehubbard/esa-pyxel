#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""
import logging
import operator
import typing as t
from copy import deepcopy
from numbers import Number

import numpy as np

from pyxel.evaluator import eval_entry
from pyxel.pipelines import DetectionPipeline, ModelGroup
from pyxel.state import get_obj_att
from pyxel.util.memory import get_size

if t.TYPE_CHECKING:
    from pyxel.detectors import CCD, CMOS


# TODO: Is this class needed ?
class Processor:
    """TBW."""

    def __init__(self, detector: t.Union["CCD", "CMOS"], pipeline: DetectionPipeline):
        self._log = logging.getLogger(__name__)

        self.detector = detector
        self.pipeline = pipeline

        self._numbytes = 0

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__  # type: str
        return f"{cls_name}<detector={self.detector!r}, pipeline={self.pipeline!r}>"

    def __deepcopy__(self, memodict: dict) -> "Processor":
        return Processor(
            detector=deepcopy(self.detector, memo=memodict),
            pipeline=deepcopy(self.pipeline, memo=memodict),
        )

    # TODO: Could it be renamed '__contains__' ?
    # TODO: reimplement this method.
    def has(self, key: str) -> bool:
        """TBW.

        Parameters
        ----------
        key : str

        Returns
        -------
        bool
            TBW.
        """
        found = False
        obj, att = get_obj_att(self, key)
        if isinstance(obj, dict) and att in obj:
            found = True
        elif hasattr(obj, att):
            found = True
        return found

    # TODO: Could it be renamed '__getitem__' ?
    # TODO: Is it really needed ?
    # TODO: What are the valid keys ? (e.g. 'detector.image.array',
    #       'detector.signal.array' and 'detector.pixel.array')
    def get(self, key: str) -> np.ndarray:
        """TBW.

        Parameters
        ----------
        key : str

        Returns
        -------
        TBW.
        """
        # return get_value(self, key)

        func = operator.attrgetter(key)  # type: t.Callable
        result = func(self)

        return np.asarray(result, dtype=np.float)

    # TODO: Could it be renamed '__setitem__' ?
    def set(
        self,
        key: str,
        value: t.Union[
            str, Number, np.ndarray, t.Sequence[t.Union[str, Number, np.ndarray]]
        ],
        convert_value: bool = True,
    ) -> None:
        """TBW.

        Parameters
        ----------
        key : str
        value
        convert_value : bool
        """
        if convert_value:  # and value:
            # TODO: Refactor this
            # convert the string based value to a number
            if isinstance(value, list):
                for i, val in enumerate(value):
                    if val:
                        value[i] = eval_entry(val)
            else:
                value = eval_entry(value)

        obj, att = get_obj_att(self, key)

        if isinstance(obj, dict) and att in obj:
            obj[att] = value
        else:
            setattr(obj, att, value)

    # TODO: Create a method `DetectionPipeline.run`
    def run_pipeline(self, abort_before: t.Optional[str] = None) -> "Processor":
        """Run a pipeline with all its models in the right order.

        Parameters
        ----------
        abort_before : str
            model name, the pipeline should be aborted before this

        Returns
        -------
        Processor
            TBW.

        Notes
        -----
        The ``Processor`` instance with method '.run_pipeline' is modified.
        """
        self._log.info("Start pipeline")

        # TODO: Use with-statement to set/unset ._is_running
        self.pipeline._is_running = True

        for group_name in self.pipeline.model_group_names:
            # Get a group of models
            models_grp = getattr(
                self.pipeline, group_name
            )  # type: t.Optional[ModelGroup]

            if models_grp:
                self._log.info("Processing group: %r", group_name)

                abort_flag = models_grp.run(
                    detector=self.detector,
                    pipeline=self.pipeline,
                    abort_model=abort_before,
                )
                if abort_flag:
                    break

        self.pipeline._is_running = False

        # TODO: Is is necessary to return 'self' ??
        return self

    @property
    def numbytes(self) -> int:
        """Recursively calculates object size in bytes using Pympler library.

        Returns
        -------
        int
            Size of the object in bytes.
        """
        self._numbytes = get_size(self)
        return self._numbytes

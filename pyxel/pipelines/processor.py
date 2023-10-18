#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""
import logging
import operator
from collections.abc import Sequence
from copy import deepcopy
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, NewType, Optional, Union

import numpy as np

from pyxel import __version__
from pyxel.evaluator import eval_entry
from pyxel.pipelines import DetectionPipeline, ModelGroup
from pyxel.state import get_obj_att
from pyxel.util.memory import get_size

if TYPE_CHECKING:
    import xarray as xr

    from pyxel.detectors import Detector


# class ResultType(Enum):
#     """Result type class."""
#
#     Photon = "photon"
#     Charge = "charge"
#     Pixel = "pixel"
#     Signal = "signal"
#     Image = "image"
#     Data = "data"
#     All = "all"

ResultId = NewType("ResultId", str)


def get_result_id(name: str) -> ResultId:
    """Convert to a 'ResultId' object."""
    if name not in (
        "scene",
        "photon",
        "charge",
        "pixel",
        "signal",
        "image",
        "data",
        "all",
    ) and not name.startswith("data"):
        raise ValueError(f"Result type: {name!r} unknown !")

    return ResultId(name)


def result_keys(result_type: ResultId) -> Sequence[ResultId]:
    """Return result keys based on result type.

    Parameters
    ----------
    result_type

    Returns
    -------
    list
    """
    if result_type == "all":
        return [
            ResultId("scene"),
            ResultId("photon"),
            ResultId("charge"),
            ResultId("pixel"),
            ResultId("signal"),
            ResultId("image"),
            ResultId("data"),
        ]

    return [ResultId(result_type)]


# TODO: Is this class needed ?
class Processor:
    """TBW."""

    def __init__(self, detector: "Detector", pipeline: DetectionPipeline):
        self._log = logging.getLogger(__name__)

        self.detector = detector
        self.pipeline = pipeline
        self._result: Optional[dict] = None  # TODO: Deprecate this variable ?

        self._numbytes = 0

    def __repr__(self) -> str:
        cls_name: str = self.__class__.__name__
        return f"{cls_name}<detector={self.detector!r}, pipeline={self.pipeline!r}>"

    def __deepcopy__(self, memodict: dict) -> "Processor":
        return Processor(
            detector=deepcopy(self.detector, memo=memodict),
            pipeline=deepcopy(self.pipeline, memo=memodict),
        )

    # TODO: Could it be renamed '__contains__' ?
    # TODO: reimplement this method.
    def has(self, key: str) -> bool:
        """Check if a parameter is available in this Processor.

        Examples
        --------
        >>> processor = Processor(...)
        >>> processor.has("pipeline.photon_collection.illumination.arguments.level")
        True
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
    def get(self, key: str) -> Any:
        """Get the current value from a Parameter.

        Examples
        --------
        >>> processor = Processor()
        >>> processor.get("pipeline.photon_collection.illumination.arguments.level")
        array(0.)
        >>> processor.get("pipeline.characteristics.quantum_efficiency")
        array(0.1)
        """
        func: Callable = operator.attrgetter(key)
        result = func(self)

        return result

    # TODO: Could it be renamed '__setitem__' ?
    def set(  # noqa: A003
        self,
        key: str,
        value: Union[str, Number, np.ndarray, list[Union[str, Number, np.ndarray]]],
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
                new_value_lst: list[Union[str, Number, np.ndarray]] = []

                val: Union[str, Number, np.ndarray]
                for val in value:
                    new_val = eval_entry(val) if val else val
                    new_value_lst.append(new_val)

                new_value: Union[
                    str, Number, np.ndarray, list[Union[str, Number, np.ndarray]]
                ] = new_value_lst

            else:
                converted_value: Union[str, Number, np.ndarray] = eval_entry(value)

                new_value = converted_value
        else:
            new_value = value

        obj, att = get_obj_att(self, key)

        if isinstance(obj, dict) and att in obj:
            obj[att] = new_value
        else:
            setattr(obj, att, new_value)

    # TODO: Create a method `DetectionPipeline.run`
    def run_pipeline(self, with_intermediate_steps: bool = False) -> None:
        """Run a pipeline with all its models in the right order.

        Parameters
        ----------
        with_intermediate_steps : bool

        Notes
        -----
        The ``Processor`` instance with method '.run_pipeline' is modified.
        """
        self._log.info("Start pipeline")

        for group_name in self.pipeline.model_group_names:
            # Get a group of models
            models_grp: Optional[ModelGroup] = getattr(self.pipeline, group_name)
            if not models_grp:
                continue

            self._log.info("Processing group: %r", group_name)
            models_grp.run(
                detector=self.detector,
                with_intermediate_steps=with_intermediate_steps,
            )

    # TODO: Refactor '.result'. See #524. Deprecate this method ?
    @property
    def result(self) -> dict:
        """Return exposure pipeline final result in a dictionary."""
        if not self._result:
            raise ValueError("No result saved in the processor.")
        return self._result

    # TODO: Refactor '.result'. See #524. Deprecate this method ?
    @result.setter
    def result(self, result_to_save: dict) -> None:
        """Set result."""
        self._result = result_to_save

    # TODO: Refactor '.result'. See #524
    def result_to_dataset(
        self,
        y: range,
        x: range,
        times: np.ndarray,
        result_type: ResultId,
    ) -> "xr.Dataset":
        """Return the result in a xarray dataset."""
        # Late import to speedup start-up time
        import xarray as xr

        if not self._result:
            raise ValueError("No result saved in the processor.")

        readout_time = xr.DataArray(
            times,
            dims=("readout_time",),
            attrs={"units": "s", "standard_name": "Readout time"},
        )

        lst: list[xr.DataArray] = []

        key: ResultId
        for key in result_keys(result_type):
            if key.startswith("data") or key.startswith("scene"):
                continue
            elif key == "photon":
                standard_name = "Photon"
                unit = "photon"
            elif key == "charge":
                standard_name = "Charge"
                unit = "electron"
            elif key == "pixel":
                standard_name = "Pixel"
                unit = "electron"
            elif key == "signal":
                standard_name = "Signal"
                unit = "volt"
            elif key == "image":
                standard_name = "Image"
                unit = "adu"
            else:
                raise NotImplementedError
                # standard_name = key
                # unit = ""

            da = xr.DataArray(
                self.result[key],
                dims=("readout_time", "y", "x"),
                name=key,
                coords={"readout_time": readout_time, "y": y, "x": x},
                attrs={"units": unit, "standard_name": standard_name},
            )

            lst.append(da)

        ds: xr.Dataset = xr.merge(lst, combine_attrs="drop_conflicts")
        ds.attrs.update({"pyxel version": __version__})

        return ds

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

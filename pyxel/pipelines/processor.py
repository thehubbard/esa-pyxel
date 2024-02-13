#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""
import logging
import operator
import warnings
from collections.abc import Sequence
from copy import deepcopy
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, NewType, Optional, Union

import numpy as np

from pyxel import __version__
from pyxel.evaluator import eval_entry
from pyxel.pipelines import DetectionPipeline, ModelGroup
from pyxel.util import get_size

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


# TODO: Refactor this function (e.g. include it in 'Processor')
def _get_obj_att(
    obj: Any, key: str, obj_type: Optional[type] = None
) -> tuple[Any, str]:
    """Retrieve an object associated with a specified key.

    The function is versatile and can be applied to dictionaries, lists,
    or user-defined objects, such as configuration models.

    Parameters
    ----------
    obj : Any
        The target object from which to extract the desired attribute.
    key : str
        A string representing the attribute path within the object.
        Nested attributes are separated by dots.
    obj_type
        An optional parameter specifying the expected type of the retrieved object.
        If provided, the function ensures that the final object matches this type.

    Returns
    -------
    A tuple containing two elements

    1. The object associated with the specified key.
    2. The name of the attribute, extracted from the key.

    Examples
    --------
    >>> obj = {"processor": {"pipeline": {"models": [1, 2, 3]}}}
    >>> get_obj_att(obj, "processor.pipeline.models")
    ({'models': [1, 2, 3]}, 'models')
    """
    *body, tail = key.split(".")
    for part in body:
        try:
            if isinstance(obj, dict):
                obj = obj[part]
            elif isinstance(obj, list):
                try:
                    index = int(part)
                    obj = obj[index]
                except ValueError:
                    for _, obj_i in enumerate(obj):
                        if hasattr(obj_i, part):
                            obj = getattr(obj_i, part)
                            break
                        elif obj_i.__class__.__name__ == part:
                            if hasattr(obj_i, tail):
                                obj = obj_i
                                break
            elif hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                raise NotImplementedError(
                    f"obj={obj!r}, key={key!r}, obj_type={obj_type!r}, part={part!r}"
                )

            if obj_type and isinstance(obj, obj_type):
                return obj, tail

        except AttributeError:
            # logging.error('Cannot find attribute %r in key %r', part, key)
            obj = None
            break
    return obj, tail


# TODO: Is this class needed ?
class Processor:
    """Represent a processor that execute pipeline.

    It manages the execution of models in the pipeline.

    Parameters
    ----------
    detector : Detector
        The detector object associated with the processor.
    pipeline : DetectionPipeline
        The detection pipeline object defining the sequence of model groups.
    """

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
        obj, att = _get_obj_att(self, key)
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
    def set(
        self,
        key: str,
        value: Union[str, Number, np.ndarray, Sequence[Union[str, Number, np.ndarray]]],
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
                    str, Number, np.ndarray, Sequence[Union[str, Number, np.ndarray]]
                ] = new_value_lst

            elif isinstance(value, (str, Number, np.ndarray)):
                converted_value: Union[str, Number, np.ndarray] = eval_entry(value)

                new_value = converted_value

            else:
                raise TypeError

        else:
            new_value = value

        obj, att = _get_obj_att(self, key)

        if isinstance(obj, dict) and att in obj:
            obj[att] = new_value
        else:
            setattr(obj, att, new_value)

    # TODO: Create a method `DetectionPipeline.run`
    def run_pipeline(self, debug: bool) -> None:
        """Run a pipeline with all its models in the right order.

        Parameters
        ----------
        debug : bool

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
                debug=debug,
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
        warnings.warn(
            "Deprecated. Will be removed in Pyxel 2.0", DeprecationWarning, stacklevel=1
        )

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

            if key == "photon":
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

            if key not in self.result:
                continue

            # TODO: 'self.result' returns a numpy array, it should returns an xarray DataArray
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

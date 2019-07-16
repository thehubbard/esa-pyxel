"""TBW."""
import typing as t
from esapy_config import get_obj_att, eval_entry, get_value
from pyxel.detectors.ccd import CCD
from pyxel.detectors.cmos import CMOS
from pyxel.pipelines.pipeline import DetectionPipeline
# from pyxel.pipelines.ccd_pipeline import CCDDetectionPipeline
# from pyxel.pipelines.cmos_pipeline import CMOSDetectionPipeline


# FRED: Is this class needed ?
class Processor:
    """TBW."""

    def __init__(self,
                 detector: t.Union[CCD, CMOS],
                 pipeline: DetectionPipeline):
        """TBW.

        :param detector:
        :param pipeline:
        """
        self.detector = detector
        self.pipeline = pipeline

    # FRED: Could it be renamed '__contains__' ?
    def has(self, key: str) -> bool:
        """TBW.

        :param key:
        :return:
        """
        found = False
        obj, att = get_obj_att(self, key)
        if isinstance(obj, dict) and att in obj:
            found = True
        elif hasattr(obj, att):
            found = True
        return found

    # FRED: Could it be renamed '__getitem__' ?
    def get(self, key: str) -> t.Any:
        """TBW.

        :param key:
        :return:
        """
        return get_value(self, key)

    # FRED: Could it be renamed '__setitem__' ?
    def set(self, key: str, value: t.Any, convert_value: bool = True) -> None:
        """TBW.

        :param key:
        :param value:
        :param convert_value:
        :return:
        """
        if convert_value:  # and value:
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

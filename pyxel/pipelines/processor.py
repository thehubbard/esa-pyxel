"""TBW."""
import typing as t

from pyxel.detectors.ccd import CCD
from pyxel.detectors.cmos import CMOS
from pyxel.evaluator import eval_entry
from pyxel.pipelines.pipeline import DetectionPipeline, ModelGroup
from pyxel.state import get_obj_att, get_value


# TODO: Is this class needed ?
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

    # def __getstate__(self):
    #     """TBW."""
    #     return {
    #         'detector': self.detector,
    #         'pipeline': self.pipeline,
    #     }

    # TODO: Could it be renamed '__contains__' ?
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

    # TODO: Could it be renamed '__getitem__' ?
    def get(self, key: str) -> t.Any:
        """TBW.

        :param key:
        :return:
        """
        return get_value(self, key)

    # TODO: Could it be renamed '__setitem__' ?
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

    def run_pipeline(self, abort_before: t.Optional[str] = None) -> "Processor":
        """TBW.

        :param abort_before: str, model name, the pipeline should be aborted before this
        :return:
        """
        self.pipeline._is_running = True
        for group_name in self.pipeline.model_group_names:
            models_grp = getattr(self.pipeline, group_name)  # type: ModelGroup
            if models_grp:
                abort_flag = models_grp.run(detector=self.detector,
                                            pipeline=self.pipeline,
                                            abort_model=abort_before)
                if abort_flag:
                    break
        self.pipeline._is_running = False

        # TODO: Is is necessary to return 'self' ??
        return self

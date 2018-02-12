import typing as t

from pyxel.detectors.ccd import CCD
from pyxel.detectors.cmos import CMOS
from pyxel.pipelines.ccd_pipeline import CCDDetectionPipeline
from pyxel.pipelines.cmos_pipeline import CMOSDetectionPipeline


class Processor:

    def __init__(self,
                 detector: t.Union[CCD, CMOS],
                 pipeline: t.Union[CCDDetectionPipeline, CMOSDetectionPipeline]) -> None:
        self.detector = detector
        self.pipeline = pipeline

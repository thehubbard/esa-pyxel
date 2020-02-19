#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""
import typing as t  # noqa: F401
from copy import deepcopy

from pyxel import util
from pyxel.pipelines.model_function import ModelFunction

if t.TYPE_CHECKING:
    from ..detectors import Detector
    from ..pipelines.pipeline import DetectionPipeline


# TODO: These methods could also be as a `abc.Sequence` with magical methods:
#       __getitem__, __iter__, __len__, __contains__, ...
class ModelGroup:
    """TBW."""

    def __init__(self, models: t.List[ModelFunction]):
        """TBW.

        :param models:
        """
        self.models = models  # type: t.List[ModelFunction]

    def __deepcopy__(self, memo) -> "ModelGroup":
        """TBW."""
        copied_models = deepcopy(self.models)
        return ModelGroup(models=copied_models)

    # TODO: Why is this method returning a `bool` ?
    def run(
        self,
        detector: "Detector",
        pipeline: "DetectionPipeline",
        abort_model: t.Optional[str] = None,
    ) -> bool:
        """Execute each enabled model in this group."""
        for model in self.models:
            if model.name == abort_model:
                return True
            if not pipeline.is_running:
                raise util.PipelineAborted("Pipeline has been aborted.")
            else:
                if model.enabled:
                    # TODO: Display here information about the executed model
                    model.function(detector)
        return False

    # TODO: Is it needed ? if yes then you could also implement
    #       the magic method '__dir__'. It is useful for auto-completion
    def __getattr__(self, item: str) -> t.Optional[ModelFunction]:
        """TBW."""
        for model in self.models:
            if model.name == item:
                return model
        return None

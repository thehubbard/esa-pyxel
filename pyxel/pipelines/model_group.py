"""TBW."""
import typing as t  # noqa: F401
from pyxel.pipelines.model_function import ModelFunction
from pyxel import util


# FRED: Build this class with ESAPY_EGSE.
class ModelGroup:
    """TBW."""

    def __init__(self, models: t.List[ModelFunction]) -> None:
        """TBW.

        :param models:
        """
        self.models = models    # type: t.List[ModelFunction]

    # FRED: Why is this method returning a `bool` ?
    def run(self, detector, pipeline, abort_model: str = None):
        """Execute each enabled model in this group."""
        for model in self.models:
            if model.name == abort_model:
                return True
            if not pipeline.is_running:
                raise util.PipelineAborted('Pipeline has been aborted.')
            else:
                if model.enabled:
                    model.function(detector)
        return False

    # FRED: Is is needed ?
    def __getstate__(self):
        """TBW."""
        return {'models': self.models}

    def __setstate__(self, item):
        """TBW."""
        self.models = item['models']

    # FRED: These methods could also be implemented:
    #       __getitem__, __iter__, __len__, __contains__, __eq__, ...

    # FRED: Is it needed ? if yes then you could also implement
    #       the magic method '__dir__'. It is useful for auto-completion
    def __getattr__(self, item):
        """TBW."""
        for model in self.models:
            if model.name == item:
                return model
        return None

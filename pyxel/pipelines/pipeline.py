"""TBW."""
# import logging
import typing as t  # noqa: F401
from pyxel.pipelines.model_group import ModelGroup
from pyxel.detectors.detector import Detector


# FRED: Add more and better typing information
class DetectionPipeline:
    """TBW."""

    # HANS: develop a ModelGroupList class. Pass this as a single argument.
    # HANS: remove _model_groups order list.
    def __init__(self,      # TODO: Too many instance attributes
                 photon_generation: t.Optional[ModelGroup] = None,
                 optics: t.Optional[ModelGroup] = None,
                 charge_generation: t.Optional[ModelGroup] = None,
                 charge_collection: t.Optional[ModelGroup] = None,
                 charge_transfer: t.Optional[ModelGroup] = None,
                 charge_measurement: t.Optional[ModelGroup] = None,
                 signal_transfer: t.Optional[ModelGroup] = None,
                 readout_electronics: t.Optional[ModelGroup] = None,
                 _model_groups: t.Optional[t.List[str]] = None,                        # TODO
                 doc: t.Optional[str] = None):
        """TBW.

        :param photon_generation:
        :param optics:
        :param charge_generation:
        :param charge_collection:
        :param charge_transfer:
        :param charge_measurement:
        :param signal_transfer:
        :param readout_electronics:
        :param _model_groups:
        :param doc:
        """
        self._is_running = False
        self.doc = doc

        self.photon_generation = photon_generation              # type: t.Optional[ModelGroup]
        self.optics = optics                                    # type: t.Optional[ModelGroup]
        self.charge_generation = charge_generation              # type: t.Optional[ModelGroup]
        self.charge_collection = charge_collection              # type: t.Optional[ModelGroup]
        self.charge_measurement = charge_measurement            # type: t.Optional[ModelGroup]
        self.readout_electronics = readout_electronics          # type: t.Optional[ModelGroup]
        self.charge_transfer = charge_transfer                  # type: t.Optional[ModelGroup]  # CCD
        self.signal_transfer = signal_transfer                  # type: t.Optional[ModelGroup]  # CMOS

        # HANS: this defines the order of steps in the pipeline. The ModelGroupList does this. Is it really needed?
        # FRED: if this is needed then it should be immutable (=> it should be a `Tuple`)
        self._model_groups = ['photon_generation',
                              'optics',
                              'charge_generation',
                              'charge_collection',
                              'charge_transfer',
                              'charge_measurement',
                              'signal_transfer',
                              'readout_electronics']            # type: t.List[str]           # TODO

    @property
    def model_group_names(self) -> t.List[str]:
        """TBW."""
        return self._model_groups

    @property
    def is_running(self) -> bool:
        """Return the running state of this pipeline."""
        return self._is_running

    def abort(self) -> None:
        """TBW."""
        self._is_running = False

    def get_model(self, name: str) -> ModelGroup:
        """TBW.

        :param name:
        :return:
        """
        for group_name in self.model_group_names:
            # FRED: This should be refactored (e.g. provide directly a `ModelGroup` and not
            #       its name)
            model_group = getattr(self, group_name)     # type: ModelGroup
            if model_group:
                for model in model_group.models:
                    if name == model.name:
                        return model

        raise AttributeError('Model has not found.')

    # FRED: In this function, the input parameter 'detector' is modified.
    #       It would be maybe better to do in this function:
    #          1. A deep copy of `detector`
    #          2. And use this copy for all processing.
    #        In this case the output `Detector` instance is different than the input `Detector`
    def run_pipeline(self, detector: Detector, abort_before: t.Optional[str] = None) -> Detector:
        """TBW.

        :param detector:
        :param abort_before: str, model name, the pipeline should be aborted before this
        :return:
        """
        self._is_running = True
        for group_name in self.model_group_names:
            # FRED: This should be refactored (e.g. provide directly a `ModelGroup` and not
            #       its name)
            models_grp = getattr(self, group_name)      # type: ModelGroup
            if models_grp:
                # HANS: it may be better to have `run` raise an Abort exception
                #     and that it is caught here. Working with flag switches is brittle.
                abort_flag = models_grp.run(detector, self, abort_model=abort_before)
                if abort_flag:
                    break
        self._is_running = False
        return detector

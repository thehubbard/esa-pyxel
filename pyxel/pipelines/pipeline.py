"""TBW."""
# import logging
import typing as t  # noqa: F401
from pyxel.pipelines.model_group import ModelGroup


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

    # def __getstate__(self):
    #     """TBW."""
    #     return {
    #         'photon_generation': self.photon_generation,
    #         'optics': self.optics,
    #         'charge_generation': self.charge_generation,
    #         'charge_collection': self.charge_collection,
    #         'charge_transfer': self.charge_transfer,
    #         'charge_measurement': self.charge_measurement,
    #         'signal_transfer': self.signal_transfer,
    #         'readout_electronics': self.readout_electronics,
    #         '_model_groups': self.model_group_names,              # TODO
    #     }

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

    def get_model(self, name: str):
        """TBW.

        :param name:
        :return:
        """
        for group_name in self.model_group_names:
            model_group = getattr(self, group_name)     # type: ModelGroup
            if model_group:
                for model in model_group.models:
                    if name == model.name:
                        return model
        raise AttributeError('Model has not found.')

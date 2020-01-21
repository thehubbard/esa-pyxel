#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""
import typing as t

from pyxel.pipelines.model_function import ModelFunction
from pyxel.pipelines.model_group import ModelGroup


class DetectionPipeline:
    """TBW."""

    # TODO: develop a ModelGroupList class ? Pass this as a single argument.
    def __init__(
        self,  # TODO: Too many instance attributes
        photon_generation: t.Optional[ModelGroup] = None,
        optics: t.Optional[ModelGroup] = None,
        charge_generation: t.Optional[ModelGroup] = None,
        charge_collection: t.Optional[ModelGroup] = None,
        charge_transfer: t.Optional[ModelGroup] = None,
        charge_measurement: t.Optional[ModelGroup] = None,
        signal_transfer: t.Optional[ModelGroup] = None,
        readout_electronics: t.Optional[ModelGroup] = None,
        doc: t.Optional[str] = None,
    ):
        """TBW.

        :param photon_generation:
        :param optics:
        :param charge_generation:
        :param charge_collection:
        :param charge_transfer:
        :param charge_measurement:
        :param signal_transfer:
        :param readout_electronics:
        :param doc:
        """
        self._is_running = False
        self.doc = doc

        self.photon_generation = photon_generation  # type: t.Optional[ModelGroup]
        self.optics = optics  # type: t.Optional[ModelGroup]
        self.charge_generation = charge_generation  # type: t.Optional[ModelGroup]
        self.charge_collection = charge_collection  # type: t.Optional[ModelGroup]
        self.charge_measurement = charge_measurement  # type: t.Optional[ModelGroup]
        self.readout_electronics = readout_electronics  # type: t.Optional[ModelGroup]
        self.charge_transfer = charge_transfer  # type: t.Optional[ModelGroup]  # CCD
        self.signal_transfer = signal_transfer  # type: t.Optional[ModelGroup]  # CMOS

        # TODO: this defines the order of steps in the pipeline.
        #       The ModelGroupList could do this. Is it really needed?
        self.MODEL_GROUPS = (
            "photon_generation",
            "optics",
            "charge_generation",
            "charge_collection",
            "charge_transfer",
            "charge_measurement",
            "signal_transfer",
            "readout_electronics",
        )  # type: t.Tuple[str, ...]

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
    def model_group_names(self) -> t.Tuple[str, ...]:
        """TBW."""
        return self.MODEL_GROUPS

    @property
    def is_running(self) -> bool:
        """Return the running state of this pipeline."""
        return self._is_running

    def abort(self) -> None:
        """TBW."""
        self._is_running = False

    def get_model(self, name: str) -> ModelFunction:
        """TBW.

        :param name:
        :return:
        """
        for group_name in self.model_group_names:  # type: str
            model_group = getattr(self, group_name)  # type: ModelGroup
            if model_group:
                for model in model_group.models:  # type: ModelFunction
                    if name == model.name:
                        return model
        raise AttributeError("Model has not found.")

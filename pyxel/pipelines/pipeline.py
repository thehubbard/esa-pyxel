#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""

import warnings
from collections.abc import Iterable, Sequence
from typing import Optional

from pyxel.pipelines import ModelFunction, ModelGroup


class DetectionPipeline:
    """TBW."""

    # Define the order of steps in the pipeline.
    MODEL_GROUPS: tuple[str, ...] = (
        "scene_generation",
        "photon_collection",
        "photon_generation",  # Deprecated. It will be removed in version 2.0
        "optics",  # Deprecated. It will be removed in version 2.0
        "phasing",
        "charge_generation",
        "charge_collection",
        "charge_transfer",
        "charge_measurement",
        "signal_transfer",
        "readout_electronics",
        "data_processing",
    )

    # TODO: develop a ModelGroupList class ? See #333
    def __init__(
        self,  # TODO: Too many instance attributes
        scene_generation: Optional[Sequence[ModelFunction]] = None,
        photon_collection: Optional[Sequence[ModelFunction]] = None,
        photon_generation: Optional[Sequence[ModelFunction]] = None,  # Deprecated
        optics: Optional[Sequence[ModelFunction]] = None,  # Deprecated
        phasing: Optional[Sequence[ModelFunction]] = None,
        charge_generation: Optional[Sequence[ModelFunction]] = None,
        charge_collection: Optional[Sequence[ModelFunction]] = None,
        charge_transfer: Optional[Sequence[ModelFunction]] = None,
        charge_measurement: Optional[Sequence[ModelFunction]] = None,
        signal_transfer: Optional[Sequence[ModelFunction]] = None,
        readout_electronics: Optional[Sequence[ModelFunction]] = None,
        data_processing: Optional[Sequence[ModelFunction]] = None,
        doc: Optional[str] = None,
    ):
        self.doc = doc

        self._scene_generation: Optional[ModelGroup] = (
            ModelGroup(scene_generation, name="scene_generation")
            if scene_generation
            else None
        )

        self._photon_collection: Optional[ModelGroup] = (
            ModelGroup(photon_collection, name="photon_collection")
            if photon_collection
            else None
        )

        # Deprecated
        self._photon_generation: Optional[ModelGroup] = (
            ModelGroup(photon_generation, name="photon_generation")
            if photon_generation
            else None
        )

        # Deprecated
        self._optics: Optional[ModelGroup] = (
            ModelGroup(optics, name="optics") if optics else None
        )

        self._phasing: Optional[ModelGroup] = (
            ModelGroup(phasing, name="phasing") if phasing else None
        )  # MKID-array

        self._charge_generation: Optional[ModelGroup] = (
            ModelGroup(charge_generation, name="charge_generation")
            if charge_generation
            else None
        )

        self._charge_collection: Optional[ModelGroup] = (
            ModelGroup(charge_collection, name="charge_collection")
            if charge_collection
            else None
        )

        self._charge_measurement: Optional[ModelGroup] = (
            ModelGroup(charge_measurement, name="charge_measurement")
            if charge_measurement
            else None
        )

        self._readout_electronics: Optional[ModelGroup] = (
            ModelGroup(readout_electronics, name="readout_electronics")
            if readout_electronics
            else None
        )

        self._charge_transfer: Optional[ModelGroup] = (
            ModelGroup(charge_transfer, name="charge_transfer")
            if charge_transfer
            else None
        )  # CCD

        self._signal_transfer: Optional[ModelGroup] = (
            ModelGroup(signal_transfer, name="signal_transfer")
            if signal_transfer
            else None
        )  # CMOS

        self._data_processing: Optional[ModelGroup] = (
            ModelGroup(data_processing, name="data_processing")
            if data_processing
            else None
        )

    def __repr__(self) -> str:
        cls_name: str = self.__class__.__name__
        return f"{cls_name}<doc={self.doc!r}>"

    def __iter__(self) -> Iterable[ModelFunction]:
        for model in self.MODEL_GROUPS:
            models_grp: Optional[ModelGroup] = getattr(self, model)
            if models_grp:
                yield from models_grp

    @property
    def scene_generation(self) -> Optional[ModelGroup]:
        """Get group 'scene generation'."""
        return self._scene_generation

    @property
    def photon_collection(self) -> Optional[ModelGroup]:
        """Get group 'photon collection'."""
        return self._photon_collection

    @property
    def photon_generation(self) -> Optional[ModelGroup]:
        """Get group 'photon generation'."""
        warnings.warn(
            "Group 'photon_generation' is deprecated "
            "and will be removed in version 2.0. "
            "Use group 'photon_collection' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._photon_generation

    @property
    def optics(self) -> Optional[ModelGroup]:
        """Get group 'optics'."""
        warnings.warn(
            "Group 'optics' is deprecated "
            "and will be removed in version 2.0. "
            "Use group 'photon_collection' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._optics

    @property
    def phasing(self) -> Optional[ModelGroup]:
        """Get group 'phasing'."""
        return self._phasing

    @property
    def charge_generation(self) -> Optional[ModelGroup]:
        """Get group 'charge generation'."""
        return self._charge_generation

    @property
    def charge_collection(self) -> Optional[ModelGroup]:
        """Get group 'charge collection'."""
        return self._charge_collection

    @property
    def charge_transfer(self) -> Optional[ModelGroup]:
        """Get group 'charge transfer'."""
        return self._charge_transfer

    @property
    def charge_measurement(self) -> Optional[ModelGroup]:
        """Get group 'charge measurement'."""
        return self._charge_measurement

    @property
    def signal_transfer(self) -> Optional[ModelGroup]:
        """Get group 'signal transfer'."""
        return self._signal_transfer

    @property
    def readout_electronics(self) -> Optional[ModelGroup]:
        """Get group 'readout electronics'."""
        return self._readout_electronics

    @property
    def data_processing(self) -> Optional[ModelGroup]:
        """Get group 'data processing'."""
        return self._data_processing

    @property
    def model_group_names(self) -> tuple[str, ...]:
        """TBW."""
        return self.MODEL_GROUPS

    # TODO: Is this method used ?
    def get_model(self, name: str) -> ModelFunction:
        """Return a ModelFunction object for the specified model name.

        Parameters
        ----------
        name: str
            Name of the model.

        Returns
        -------
        ModelFunction

        Raises
        ------
        AttributeError
            If model with the specified name is not found.
        """
        group_name: str
        for group_name in self.model_group_names:
            model_group: ModelGroup = getattr(self, group_name)
            if model_group:
                model: ModelFunction
                for model in model_group.models:
                    if name == model.name:
                        return model
        raise AttributeError("Model has not been found.")

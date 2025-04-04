#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
"""Configuration loader."""

import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import IO, TYPE_CHECKING, Any, Optional, Union

from pyxel import __version__ as version
from pyxel.detectors import (
    APD,
    CCD,
    CMOS,
    MKID,
    APDCharacteristics,
    APDGeometry,
    CCDGeometry,
    Characteristics,
    CMOSGeometry,
    Environment,
    MKIDGeometry,
)
from pyxel.exposure import Exposure, Readout
from pyxel.observation import Observation, ParameterValues
from pyxel.outputs import CalibrationOutputs, ExposureOutputs, ObservationOutputs
from pyxel.pipelines import DetectionPipeline, FitnessFunction, ModelFunction

if TYPE_CHECKING:
    from pyxel.calibration import Algorithm, Calibration


@dataclass
class Configuration:
    """Configuration class."""

    pipeline: DetectionPipeline

    # Running modes
    exposure: Exposure | None = None
    observation: Observation | None = None
    calibration: Optional["Calibration"] = None

    # Detectors
    ccd_detector: CCD | None = None
    cmos_detector: CMOS | None = None
    mkid_detector: MKID | None = None
    apd_detector: APD | None = None

    def __post_init__(self):
        # Sanity checks
        running_modes = [self.exposure, self.observation, self.calibration]
        num_running_modes: int = sum(el is not None for el in running_modes)

        if num_running_modes != 1:
            raise ValueError(
                "Expecting only one running mode: "
                "'exposure', 'observation' or 'calibration'."
            )

        detectors = [
            self.ccd_detector,
            self.cmos_detector,
            self.mkid_detector,
            self.apd_detector,
        ]
        num_detectors = sum(el is not None for el in detectors)

        if num_detectors != 1:
            raise ValueError(
                "Expecting only one detector: 'ccd_detector', 'cmos_detector',"
                " 'mkid_detector' or 'apd_detector'."
            )

    def __repr__(self) -> str:
        cls_name: str = self.__class__.__name__

        params: list[str] = [f"pipeline={self.pipeline!r}"]

        # Get mode
        if self.exposure is not None:
            params.append(f"exposure={self.exposure!r}")
        elif self.observation is not None:
            params.append(f"observation={self.observation!r}")
        elif self.calibration is not None:
            params.append(f"calibration={self.calibration!r}")
        else:
            # Do nothing
            pass

        # Get detector
        if self.ccd_detector is not None:
            params.append(f"ccd_detector={self.ccd_detector!r}")
        elif self.cmos_detector is not None:
            params.append(f"cmos_detector={self.cmos_detector!r}")
        elif self.mkid_detector is not None:
            params.append(f"mkid_detector={self.mkid_detector!r}")
        elif self.apd_detector is not None:
            params.append(f"apd_detector={self.apd_detector!r}")
        else:
            # Do nothing
            pass

        return f"{cls_name}({', '.join(params)})"

    @property
    def running_mode(self) -> Union[Exposure, Observation, "Calibration"]:
        """Get current running mode."""
        if self.exposure is not None:
            return self.exposure
        elif self.observation is not None:
            return self.observation
        elif self.calibration is not None:
            return self.calibration
        else:
            raise NotImplementedError

    @property
    def detector(self) -> CCD | CMOS | MKID | APD:
        """Get current detector."""
        if self.ccd_detector is not None:
            return self.ccd_detector
        elif self.cmos_detector is not None:
            return self.cmos_detector
        elif self.mkid_detector is not None:
            return self.mkid_detector
        elif self.apd_detector is not None:
            return self.apd_detector
        else:
            raise NotImplementedError


def load(yaml_file: str | Path) -> Configuration:
    """Load configuration from a ``YAML`` file."""
    filename = Path(yaml_file).resolve()
    if not filename.exists():
        raise FileNotFoundError(f"Cannot find configuration file '{filename}'.")
    with filename.open("r") as file_obj:
        dct = load_yaml(file_obj)

    return _build_configuration(dct)


def loads(yaml_string: str) -> Configuration:
    """Load configuration from a ``YAML`` string."""
    dct = load_yaml(yaml_string)
    return _build_configuration(dct)


def load_yaml(stream: str | IO) -> Any:
    """Load a ``YAML`` document."""
    # Late import to speedup start-up time
    import yaml

    result = yaml.load(stream, Loader=yaml.SafeLoader)
    return result


def to_exposure_outputs(dct: dict) -> ExposureOutputs:
    """Create a ExposureOutputs class from a dictionary."""
    return ExposureOutputs(**dct)


def to_readout(dct: dict | None = None) -> Readout:
    """Create a Readout class from a dictionary."""
    if dct is None:
        dct = {}
    return Readout(**dct)


def to_exposure(dct: dict | None) -> Exposure:
    """Create a Exposure class from a dictionary."""
    if dct is None:
        dct = {}

    if "outputs" in dct:
        dct.update({"outputs": to_exposure_outputs(dct["outputs"])})

    dct.update({"readout": to_readout(dct.get("readout"))})

    return Exposure(**dct)


def to_observation_outputs(dct: dict | None) -> ObservationOutputs | None:
    """Create a ObservationOutputs class from a dictionary."""
    if dct is None:
        return None

    output_folder = dct["output_folder"]
    custom_dir_name = dct.get("custom_dir_name", "")
    save_data_to_file = dct.get("save_data_to_file")

    if "save_observation_data" in dct:
        warnings.warn(
            "Deprecated. Will be removed in future version",
            DeprecationWarning,
            stacklevel=1,
        )

        save_observation_data = dct.get("save_observation_data")

        return ObservationOutputs(
            output_folder=output_folder,
            custom_dir_name=custom_dir_name,
            save_data_to_file=save_data_to_file,
            save_observation_data=save_observation_data,
        )
    else:
        return ObservationOutputs(
            output_folder=output_folder,
            custom_dir_name=custom_dir_name,
            save_data_to_file=save_data_to_file,
        )


def to_parameters(dct: Mapping[str, Any]) -> ParameterValues:
    """Create a ParameterValues class from a dictionary."""
    return ParameterValues(**dct)


def to_observation(dct: dict) -> Observation:
    """Create a Parametric class from a dictionary."""
    parameters: Sequence[Mapping[str, Any]] = dct.get("parameters", [])

    if not parameters:
        raise ValueError(
            "Missing entry 'parameters' in the YAML configuration file !\n"
            "Consider adding the following YAML snippet in the configuration file:\n"
            "  parameters:\n"
            "    - key: pipeline.photon_collection.illumination.arguments.level\n"
            "      value: [1, 2, 3, 4]\n",
        )

    dct.update({"parameters": [to_parameters(param_dict) for param_dict in parameters]})

    if "outputs" in dct:
        dct.update({"outputs": to_observation_outputs(dct["outputs"])})

    dct.update({"readout": to_readout(dct.get("readout"))})

    return Observation(**dct)


def to_calibration_outputs(dct: dict) -> CalibrationOutputs:
    """Create a CalibrationOutputs class from a dictionary."""
    return CalibrationOutputs(**dct)


def to_algorithm(dct: dict) -> "Algorithm":
    """Create an Algorithm class from a dictionary."""
    # Late import to speedup start-up time
    from pyxel.calibration import Algorithm

    return Algorithm(**dct)


def to_fitness_function(dct: dict) -> FitnessFunction:
    """Create a callable from a dictionary."""
    func: str = dct["func"]
    arguments: Mapping[str, Any] | None = dct.get("arguments")

    return FitnessFunction(func=func, arguments=arguments)


def to_calibration(dct: dict) -> "Calibration":
    """Create a Calibration class from a dictionary."""
    # Late import to speedup start-up time
    from pyxel.calibration import Calibration

    if "outputs" in dct:
        dct.update({"outputs": to_calibration_outputs(dct["outputs"])})

    dct.update({"fitness_function": to_fitness_function(dct["fitness_function"])})
    dct.update({"algorithm": to_algorithm(dct["algorithm"])})
    dct.update(
        {"parameters": [to_parameters(param_dict) for param_dict in dct["parameters"]]}
    )
    dct["result_input_arguments"] = [
        to_parameters(value) for value in dct.get("result_input_arguments", {})
    ]
    dct.update({"readout": to_readout(dct.get("readout"))})

    return Calibration(**dct)


def to_ccd_geometry(dct: dict) -> CCDGeometry:
    """Create a CCDGeometry class from a dictionary."""
    return CCDGeometry(**dct)


def to_cmos_geometry(dct: dict) -> CMOSGeometry:
    """Create a CMOSGeometry class from a dictionary."""
    return CMOSGeometry(**dct)


def to_mkid_geometry(dct: dict) -> MKIDGeometry:
    """Create a MKIDGeometry class from a dictionary."""
    return MKIDGeometry(**dct)


def to_apd_geometry(dct: dict) -> APDGeometry:
    """Create a APDGeometry class from a dictionary."""
    return APDGeometry(**dct)


def to_environment(dct: dict | None) -> Environment:
    """Create an Environment class from a dictionary."""
    if dct is None:
        dct = {}
    return Environment.from_dict(dct)


def to_ccd_characteristics(dct: dict | None) -> Characteristics:
    """Create a CCDCharacteristics class from a dictionary."""
    if dct is None:
        dct = {}
    return Characteristics(**dct)


def to_cmos_characteristics(dct: dict | None) -> Characteristics:
    """Create a CMOSCharacteristics class from a dictionary."""
    if dct is None:
        dct = {}
    return Characteristics(**dct)


def to_mkid_characteristics(dct: dict | None) -> Characteristics:
    """Create a MKIDCharacteristics class from a dictionary."""
    if dct is None:
        dct = {}
    return Characteristics(**dct)


def to_apd_characteristics(dct: dict | None) -> APDCharacteristics:
    """Create a APDCharacteristics class from a dictionary."""
    if dct is None:
        dct = {}
    return APDCharacteristics(**dct)


def to_ccd(dct: dict) -> CCD:
    """Create a CCD class from a dictionary."""
    return CCD(
        geometry=to_ccd_geometry(dct["geometry"]),
        environment=to_environment(dct["environment"]),
        characteristics=to_ccd_characteristics(dct["characteristics"]),
    )


def to_cmos(dct: dict) -> CMOS:
    """Create a :term:`CMOS` class from a dictionary."""
    return CMOS(
        geometry=to_cmos_geometry(dct["geometry"]),
        environment=to_environment(dct["environment"]),
        characteristics=to_cmos_characteristics(dct["characteristics"]),
    )


def to_mkid_array(dct: dict) -> MKID:
    """Create an MKIDarray class from a dictionary."""
    return MKID(
        geometry=to_mkid_geometry(dct["geometry"]),
        environment=to_environment(dct["environment"]),
        characteristics=to_mkid_characteristics(dct["characteristics"]),
    )


def to_apd(dct: dict) -> APD:
    """Create an APDarray class from a dictionary."""
    return APD(
        geometry=to_apd_geometry(dct["geometry"]),
        environment=to_environment(dct["environment"]),
        characteristics=to_apd_characteristics(dct["characteristics"]),
    )


def to_model_function(dct: dict) -> ModelFunction:
    """Create a ModelFunction class from a dictionary."""
    return ModelFunction(**dct)


def to_pipeline(dct: dict) -> DetectionPipeline:
    """Create a DetectionPipeline class from a dictionary."""
    for model_group_name in dct:
        models_list: Sequence[dict] | None = dct[model_group_name]

        if models_list is None:
            models: Sequence[ModelFunction] | None = None
        else:
            models = [to_model_function(model_dict) for model_dict in models_list]

        dct[model_group_name] = models
    return DetectionPipeline(**dct)


def _build_configuration(dct: dict) -> Configuration:
    """Create a Configuration class from a dictionary."""
    pipeline: DetectionPipeline = to_pipeline(dct["pipeline"])

    # Sanity checks
    keys_running_mode: Sequence[str] = [
        "exposure",
        "observation",
        "calibration",
    ]
    num_running_modes: int = sum(key in dct for key in keys_running_mode)
    if num_running_modes != 1:
        keys = ", ".join(map(repr, keys_running_mode))
        raise ValueError(f"Expecting only one running mode: {keys}")

    keys_detectors: Sequence[str] = [
        "ccd_detector",
        "cmos_detector",
        "mkid_detector",
        "apd_detector",
    ]
    num_detector: int = sum(key in dct for key in keys_detectors)
    if num_detector != 1:
        keys = ", ".join(map(repr, keys_detectors))
        raise ValueError(f"Expecting only one detector: {keys}")

    running_mode: dict[str, Exposure | Observation | "Calibration"] = {}
    if "exposure" in dct:
        running_mode["exposure"] = to_exposure(dct["exposure"])
    elif "observation" in dct:
        running_mode["observation"] = to_observation(dct["observation"])
    elif "calibration" in dct:
        running_mode["calibration"] = to_calibration(dct["calibration"])
    else:
        raise ValueError("No mode configuration provided.")

    detector: dict[str, CCD | CMOS | MKID | APD] = {}
    if "ccd_detector" in dct:
        detector["ccd_detector"] = to_ccd(dct["ccd_detector"])
    elif "cmos_detector" in dct:
        detector["cmos_detector"] = to_cmos(dct["cmos_detector"])
    elif "mkid_detector" in dct:
        detector["mkid_detector"] = to_mkid_array(dct["mkid_detector"])
    elif "apd_detector" in dct:
        detector["apd_detector"] = to_apd(dct["apd_detector"])
    else:
        raise ValueError("No detector configuration provided.")

    configuration: Configuration = Configuration(
        pipeline=pipeline,
        **running_mode,  # type: ignore
        **detector,  # type: ignore
    )

    return configuration


def copy_config_file(input_filename: str | Path, output_dir: Path) -> Path:
    """Save a copy of the input ``YAML`` file to output directory.

    Parameters
    ----------
    input_filename: str or Path
    output_dir: Path

    Returns
    -------
    Path
    """

    input_file = Path(input_filename)
    copy2(input_file, output_dir)

    # TODO: sort filenames ?
    pattern: str = f"*{input_file.suffix}"
    copied_input_file_it: Iterator[Path] = output_dir.glob(pattern)
    copied_input_file: Path = next(copied_input_file_it)

    with copied_input_file.open("a") as file:
        file.write("\n#########")
        file.write(f"\n# Pyxel version: {version}")
        file.write("\n#########")

    return copied_input_file

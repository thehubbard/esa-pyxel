#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""CLI to run Pyxel."""
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import click
import dask
import pandas as pd

from pyxel import Configuration
from pyxel import __version__ as version
from pyxel import load, outputs, save
from pyxel.detectors import APD, CCD, CMOS, MKID, Detector
from pyxel.exposure import Exposure
from pyxel.observation import Observation, ObservationResult
from pyxel.pipelines import DetectionPipeline, Processor
from pyxel.util import create_model, download_examples

if TYPE_CHECKING:
    import xarray as xr
    from datatree import DataTree

    from pyxel.calibration import Calibration
    from pyxel.outputs import CalibrationOutputs, ExposureOutputs, ObservationOutputs


# TODO: This function will be deprecated (see #563)
def exposure_mode(
    exposure: "Exposure",
    detector: Detector,
    pipeline: "DetectionPipeline",
) -> "xr.Dataset":
    """Run an 'exposure' pipeline.

    For more information, see :ref:`exposure_mode`.

    Parameters
    ----------
    exposure: Exposure
    detector: Detector
    pipeline: DetectionPipeline

    Returns
    -------
    Dataset
        An multi-dimensional array database from `xarray <https://xarray.pydata.org>`_.

    Examples
    --------
    Load a configuration file

    >>> import pyxel
    >>> config = pyxel.load("configuration.yaml")
    >>> config
    Configuration(...)

    Run an exposure pipeline

    >>> dataset = pyxel.exposure_mode(
    ...     exposure=config.exposure,
    ...     detector=config.detector,
    ...     pipeline=config.pipeline,
    ... )
    >>> dataset
    <xarray.Dataset>
    Dimensions:       (readout_time: 1, y: 450, x: 450)
    Coordinates:
      * readout_time  (readout_time) int64 1
      * y             (y) int64 0 1 2 3 4 5 6 7 ... 442 443 444 445 446 447 448 449
      * x             (x) int64 0 1 2 3 4 5 6 7 ... 442 443 444 445 446 447 448 449
    Data variables:
        image         (readout_time, y, x) uint16 9475 9089 8912 ... 9226 9584 10079
        signal        (readout_time, y, x) float64 3.159 3.03 2.971 ... 3.195 3.36
        pixel         (readout_time, y, x) float64 1.053e+03 1.01e+03 ... 1.12e+03
    """

    logging.info("Mode: Exposure")

    exposure_outputs: ExposureOutputs = exposure.outputs
    detector.set_output_dir(exposure_outputs.output_dir)  # TODO: Remove this

    processor = Processor(detector=detector, pipeline=pipeline)

    result: xr.Dataset = exposure.run_exposure(processor=processor)

    if exposure_outputs.save_exposure_data:
        exposure_outputs.save_exposure_outputs(dataset=result)

    return result


def _run_exposure_mode(
    exposure: "Exposure",
    detector: Detector,
    pipeline: "DetectionPipeline",
) -> "DataTree":
    """Run an 'exposure' pipeline.

    For more information, see :ref:`exposure_mode`.

    Parameters
    ----------
    exposure: Exposure
    detector: Detector
    pipeline: DetectionPipeline

    Returns
    -------
    Dataset
        An multi-dimensional array database from `xarray <https://xarray.pydata.org>`_.

    Examples
    --------
    Load a configuration file

    >>> import pyxel
    >>> config = pyxel.load("configuration.yaml")
    >>> config
    Configuration(...)

    Run an exposure pipeline

    >>> dataset = pyxel.exposure_mode(
    ...     exposure=config.exposure,
    ...     detector=config.detector,
    ...     pipeline=config.pipeline,
    ... )
    >>> dataset
    <xarray.Dataset>
    Dimensions:       (readout_time: 1, y: 450, x: 450)
    Coordinates:
      * readout_time  (readout_time) int64 1
      * y             (y) int64 0 1 2 3 4 5 6 7 ... 442 443 444 445 446 447 448 449
      * x             (x) int64 0 1 2 3 4 5 6 7 ... 442 443 444 445 446 447 448 449
    Data variables:
        image         (readout_time, y, x) uint16 9475 9089 8912 ... 9226 9584 10079
        signal        (readout_time, y, x) float64 3.159 3.03 2.971 ... 3.195 3.36
        pixel         (readout_time, y, x) float64 1.053e+03 1.01e+03 ... 1.12e+03
    """

    logging.info("Mode: Exposure")

    exposure_outputs: ExposureOutputs = exposure.outputs
    detector.set_output_dir(exposure_outputs.output_dir)  # TODO: Remove this

    processor = Processor(detector=detector, pipeline=pipeline)

    result: DataTree = exposure.run_exposure_new(processor=processor)

    if exposure_outputs.save_exposure_data:
        exposure_outputs.save_exposure_outputs(dataset=result)

    return result


# TODO: This function will be deprecated (see #563)
def observation_mode(
    observation: "Observation",
    detector: Detector,
    pipeline: "DetectionPipeline",
) -> "ObservationResult":
    """Run an 'observation' pipeline.

    For more information, see :ref:`observation_mode`.

    Parameters
    ----------
    observation: Observation
    detector: Detector
    pipeline: DetectionPipeline

    Returns
    -------
    ObservationResult
        Result.

    Examples
    --------
    Load a configuration file

    >>> import pyxel
    >>> config = pyxel.load("configuration.yaml")
    >>> config
    Configuration(...)

    Run an observation pipeline

    >>> result = pyxel.observation_mode(
    ...     exposure=config.exposure,
    ...     detector=config.detector,
    ...     pipeline=config.pipeline,
    ... )
    >>> result
    ObservationResult(...)
    """
    logging.info("Mode: Observation")

    observation_outputs: ObservationOutputs = observation.outputs
    detector.set_output_dir(observation_outputs.output_dir)  # TODO: Remove this

    # TODO: This should be done during initializing of object `Configuration`
    # parametric_outputs.params_func(parametric)

    processor = Processor(detector=detector, pipeline=pipeline)

    result: ObservationResult = observation.run_observation(processor=processor)

    if observation_outputs.save_observation_data:
        observation_outputs.save_observation_datasets(
            result=result, mode=observation.parameter_mode
        )

    return result


# TODO: This function will be deprecated (see #563)
def calibration_mode(
    calibration: "Calibration",
    detector: Detector,
    pipeline: "DetectionPipeline",
    compute_and_save: bool = True,
) -> tuple["xr.Dataset", pd.DataFrame, pd.DataFrame, Sequence]:
    """Run a 'calibration' pipeline.

    For more information, see :ref:`calibration_mode`.

    Parameters
    ----------
    calibration: Calibration
    detector: Detector
    pipeline: DetectionPipeline
    compute_and_save: bool

    Returns
    -------
    tuple of Dataset, DataFrame, DataFrame, Sequence

    Examples
    --------
    Load a configuration file

    >>> import pyxel
    >>> config = pyxel.load("configuration.yaml")
    >>> config
    Configuration(...)

    Run a calibration pipeline

    >>> ds, processors, logs, filenames = pyxel.calibration_mode(
    ...     exposure=config.exposure,
    ...     detector=config.detector,
    ...     pipeline=config.pipeline,
    ... )

    >>> ds
    <xarray.Dataset>
    Dimensions:              (island: 2, evolution: 2, param_id: 4, id_processor: 1, readout_time: 1, y: 100, x: 100)
    Coordinates:
      * island               (island) int64 1 2
      * evolution            (evolution) int64 1 2
      * id_processor         (id_processor) int64 0
      * readout_time         (readout_time) int64 1
      * y                    (y) int64 0 1 2 3 4 5 6 7 8 ... 92 93 94 95 96 97 98 99
      * x                    (x) int64 0 1 2 3 4 5 6 7 8 ... 92 93 94 95 96 97 98 99
      * param_id             (param_id) int64 0 1 2 3
    Data variables:
        champion_fitness     (island, evolution) float64 4.759e+06 ... 4.533e+06
        champion_decision    (island, evolution, param_id) float64 0.08016 ... 7....
        champion_parameters  (island, evolution, param_id) float64 0.08016 ... 7....
        simulated_image      (island, id_processor, readout_time, y, x) uint16 48...
        simulated_signal     (island, id_processor, readout_time, y, x) float64 4...
        simulated_pixel      (island, id_processor, readout_time, y, x) float64 6...
        target               (id_processor, y, x) >f8 4.834e+03 ... 4.865e+03
    Attributes:
        num_islands:      2
        population_size:  20
        num_evolutions:   2
        generations:      5
        topology:         unconnected
        result_type:      image

    >>> processors
           island  id_processor                                          processor
    0       0             0  Delayed('apply_parameters-c5da1649-766f-4ecb-a...
    1       1             0  Delayed('apply_parameters-c16f998f-f52f-4beb-b...

    >>> logs
        num_generations  ...  global_num_generations
    0                 1  ...                       1
    1                 2  ...                       2
    2                 3  ...                       3
    3                 4  ...                       4
    4                 5  ...                       5
    ..              ...  ...                     ...
    15                1  ...                       6
    16                2  ...                       7
    17                3  ...                       8
    18                4  ...                       9
    19                5  ...                      10

    >>> filenames
    []
    """
    # Late import to speedup start-up time
    from pyxel.calibration import CalibrationResult

    logging.info("Mode: Calibration")

    calibration_outputs: CalibrationOutputs = calibration.outputs
    detector.set_output_dir(calibration_outputs.output_dir)  # TODO: Remove this

    processor = Processor(detector=detector, pipeline=pipeline)

    ds_results, df_processors, df_all_logs = calibration.run_calibration(
        processor=processor, output_dir=calibration_outputs.output_dir
    )

    # TODO: Save the processors from 'df_processors'
    # TODO: Generate plots from 'ds_results'

    # TODO: Do something with 'df_all_logs' ?

    # TODO: create 'output' object with .calibration_outputs
    # TODO: use 'fitting.get_simulated_data' ==> np.ndarray

    # geometry = processor.detector.geometry
    # calibration.post_processing(
    #     champions=champions,
    #     output=calibration_outputs,
    #     row=geometry.row,
    #     col=geometry.col,
    # )
    filenames = calibration._post_processing(
        ds=ds_results, df_processors=df_processors, output=calibration_outputs
    )

    if compute_and_save:
        computed_ds, df_processors, df_logs, filenames = dask.compute(
            ds_results, df_processors, df_all_logs, filenames
        )

        if calibration_outputs.save_calibration_data:
            calibration_outputs.save_calibration_outputs(
                dataset=computed_ds, logs=df_logs
            )
            print(f"Saved calibration outputs to {calibration_outputs.output_dir}")

        result = CalibrationResult(
            dataset=computed_ds,
            processors=df_processors,
            logs=df_logs,
            filenames=filenames,
        )

    else:
        result = CalibrationResult(
            dataset=ds_results,
            processors=df_processors,
            logs=df_all_logs,
            filenames=filenames,
        )

    return result


def _run_calibration_mode(
    calibration: "Calibration",
    detector: Detector,
    pipeline: "DetectionPipeline",
) -> "DataTree":
    logging.info("Mode: Calibration")

    calibration_outputs: CalibrationOutputs = calibration.outputs
    detector.set_output_dir(calibration_outputs.output_dir)  # TODO: Remove this

    processor = Processor(detector=detector, pipeline=pipeline)

    data_tree = calibration.run_calibration_new(
        processor=processor,
        output_dir=calibration_outputs.output_dir,
    )

    return data_tree


def _run_observation_mode(
    observation: Observation,
    detector: Detector,
    pipeline: DetectionPipeline,
) -> "DataTree":
    logging.info("Mode: Observation")

    observation_outputs: ObservationOutputs = observation.outputs
    detector.set_output_dir(observation_outputs.output_dir)  # TODO: Remove this

    processor = Processor(detector=detector, pipeline=pipeline)

    result = observation.run_observation_datatree(processor=processor)

    if observation_outputs.save_observation_data:
        raise NotImplementedError
    #     observation_outputs.save_observation_datasets(
    #         result=result, mode=observation.parameter_mode
    #     )

    return result


def run_mode(
    mode: Union[Exposure, Observation, "Calibration"],
    detector: Detector,
    pipeline: DetectionPipeline,
) -> "DataTree":
    """Run a pipeline.

    Parameters
    ----------
    mode : Exposure, Observation or Calibration
        Mode to execute.
    detector : Detector
        This object is the container for all the data used for the models.
    pipeline : DetectionPipeline
        This is the core algorithm of Pyxel. This pipeline contains all the models to run.

    Returns
    -------
    DataTree

    Raises
    ------
    TypeError
        Raised if the ``mode`` is not valid.

    Examples
    --------
    Load a configuration file

    >>> import pyxel
    >>> config = pyxel.load("configuration.yaml")
    >>> config

     Run a pipeline

    >>> data = pyxel.run_mode(
    ...     mode=config.exposure,
    ...     detector=config.detector,
    ...     pipeline=config.pipeline,
    ... )
    >>> data
    DataTree('None', parent=None)
    │   Dimensions:            (period: 5, angle: 3, trap_densities_id: 7, dim_0: 2)
    │   Coordinates:
    │     * period             (period) int64 4 8 12 16 20
    │     * angle              (angle) int64 0 10 20
    │     * trap_densities_id  (trap_densities_id) int64 0 1 2 3 4 5 6
    │       trap_densities     (trap_densities_id, dim_0) float64 nan nan ... 70.0 80.0
    │   Dimensions without coordinates: dim_0
    │   Data variables:
    │       *empty*
    │   Attributes:
    │       pyxel version:  1.8+141.gb470c395
    │       running mode:   Observation - Product
    ├── DataTree('bucket')
    │       Dimensions:            (trap_densities_id: 7, angle: 3, period: 5, time: 1,
    │                               y: 100, x: 100, dim_0: 2)
    │       Coordinates:
    │         * time               (time) float64 1.0
    │         * y                  (y) int64 0 1 2 3 4 5 6 7 8 ... 92 93 94 95 96 97 98 99
    │         * x                  (x) int64 0 1 2 3 4 5 6 7 8 ... 92 93 94 95 96 97 98 99
    │         * period             (period) int64 4 8 12 16 20
    │         * angle              (angle) int64 0 10 20
    │         * trap_densities_id  (trap_densities_id) int64 0 1 2 3 4 5 6
    │           trap_densities     (trap_densities_id, dim_0) float64 nan nan ... 70.0 80.0
    │       Dimensions without coordinates: dim_0
    │       Data variables:
    │           photon             (trap_densities_id, angle, period, time, y, x) float64 ...
    │           charge             (trap_densities_id, angle, period, time, y, x) float64 ...
    │           pixel              (trap_densities_id, angle, period, time, y, x) float64 ...
    │           signal             (trap_densities_id, angle, period, time, y, x) float64 ...
    │           image              (trap_densities_id, angle, period, time, y, x) float64 ...
    └── DataTree('data')
    """
    from pyxel.calibration import Calibration

    if isinstance(mode, Exposure):
        data_tree = _run_exposure_mode(
            exposure=mode,
            detector=detector,
            pipeline=pipeline,
        )

    elif isinstance(mode, Observation):
        data_tree = _run_observation_mode(
            observation=mode,
            detector=detector,
            pipeline=pipeline,
        )

    elif isinstance(mode, Calibration):
        data_tree = _run_calibration_mode(
            calibration=mode,
            detector=detector,
            pipeline=pipeline,
        )

    else:
        raise TypeError("Please provide a valid simulation mode !")

    return data_tree


def output_directory(configuration: Configuration) -> Path:
    """Return the output directory from the configuration.

    Parameters
    ----------
    configuration

    Returns
    -------
    output_dir
    """
    # Late import to speedup start-up time
    from pyxel.calibration import Calibration

    if isinstance(configuration.exposure, Exposure):
        output_dir = configuration.exposure.outputs.output_dir
    elif isinstance(configuration.calibration, Calibration):
        output_dir = configuration.calibration.outputs.output_dir
    elif isinstance(configuration.observation, Observation):
        output_dir = configuration.observation.outputs.output_dir
    else:
        raise (TypeError("Outputs not initialized."))
    return output_dir


def run(input_filename: Union[str, Path], random_seed: Optional[int] = None) -> None:
    """Run a YAML configuration file.

    For more information, see :ref:`running_modes`.

    Parameters
    ----------
    input_filename : str or Path
    random_seed : int, optional

    Examples
    --------
    >>> import pyxel
    >>> pyxel.run("configuration.yaml")
    """
    # Late import to speedup start-up time
    from pyxel.calibration import Calibration

    logging.info("Pyxel version %s", version)
    logging.info("Pipeline started.")

    start_time = time.time()

    configuration: Configuration = load(Path(input_filename).expanduser().resolve())

    output_dir = output_directory(configuration)

    save(input_filename=input_filename, output_dir=output_dir)

    pipeline: DetectionPipeline = configuration.pipeline
    detector: Union[CCD, CMOS, MKID, APD] = configuration.detector
    running_mode: Union[Exposure, Observation, Calibration] = configuration.running_mode

    _ = run_mode(mode=running_mode, detector=detector, pipeline=pipeline)

    logging.info("Pipeline completed.")
    logging.info("Running time: %.3f seconds", (time.time() - start_time))
    # Closing the logger in order to be able to move the file in the output dir
    logging.shutdown()
    outputs.save_log_file(output_dir)

    # Late import to speedup start-up time
    from matplotlib import pyplot as plt

    plt.close()


# TODO: Add an option to display colors ?
@click.group()
@click.version_option(version=version)
def main():
    """Pyxel detector simulation framework.

    Pyxel is a detector simulation framework, that can simulate a variety of
    detector effects (e.g., cosmic rays, radiation-induced :term:`CTI` in :term:`CCDs<CCD>`, persistence
    in :term:`MCT`, charge diffusion, crosshatches, noises, crosstalk etc.) on a given image.
    """


@main.command(name="download-examples")
@click.argument("folder", type=click.Path(), default="pyxel-examples", required=False)
@click.option("-f", "--force", is_flag=True, help="Force flag for saving the examples.")
def download_pyxel_examples(folder, force: bool):
    """Install examples to a specified directory.

    Default folder is './pyxel-examples'.
    """
    download_examples(foldername=folder, force=force)


@main.command(name="create-model")
@click.argument("model_name", type=str)
def create_new_model(model_name: str):
    """Create a new model.

    Use: arg1/arg2. Create a new module in ``pyxel/models/arg1/arg2`` using a template
    (``pyxel/templates/MODELTEMPLATE.py``)
    """
    create_model(newmodel=model_name)


@main.command(name="run")
@click.argument("config", type=click.Path(exists=True))
@click.option(
    "-v",
    "--verbosity",
    count=True,
    show_default=True,
    help="Increase output verbosity (-v/-vv/-vvv)",
)
def run_config(config: str, verbosity: int):
    """Run Pyxel with a ``YAML`` configuration file."""
    logging_level = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG][
        min(verbosity, 3)
    ]
    log_format = (
        "%(asctime)s - %(name)s - %(threadName)30s - %(funcName)30s \t %(message)s"
    )
    logging.basicConfig(
        filename="pyxel.log",
        level=logging_level,
        format=log_format,
        datefmt="%d-%m-%Y %H:%M:%S",
    )

    # If user wants the log in stdout AND in file, use the three lines below
    stream_stdout = logging.StreamHandler(sys.stdout)
    stream_stdout.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(stream_stdout)

    run(input_filename=config)


if __name__ == "__main__":
    main()

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Pyxel detector simulation framework.

Pyxel is a detector simulation framework, that can simulate a variety of
detector effects (e.g., cosmic rays, radiation-induced CTI in CCDs, persistence
in MCT, charge diffusion, crosshatches, noises, crosstalk etc.) on a given image.
"""
import argparse
import logging
import sys
import time
import typing as t
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from pyxel import __version__ as version
from pyxel import outputs
from pyxel.calibration import Calibration, CalibrationResult
from pyxel.configuration import Configuration, load, save
from pyxel.detectors import CCD, CMOS, MKID, Detector
from pyxel.exposure import Exposure
from pyxel.parametric import Parametric, ParametricResult
from pyxel.pipelines import DetectionPipeline, Processor
from pyxel.util import create_model, download_examples

if t.TYPE_CHECKING:
    from .outputs import CalibrationOutputs, ExposureOutputs, ParametricOutputs


def exposure_mode(
    exposure: "Exposure",
    detector: Detector,
    pipeline: "DetectionPipeline",
) -> xr.Dataset:
    """Run an 'exposure' pipeline.

    Parameters
    ----------
    exposure
    detector
    pipeline

    Returns
    -------
    None
    """

    logging.info("Mode: Exposure")

    exposure_outputs = exposure.outputs  # type: ExposureOutputs

    detector.set_output_dir(exposure_outputs.output_dir)  # TODO: Remove this

    processor = Processor(detector=detector, pipeline=pipeline)

    result = exposure.run_exposure(processor=processor)

    if exposure_outputs.save_exposure_data:
        exposure_outputs.save_exposure_outputs(dataset=result)

    return result


def parametric_mode(
    parametric: "Parametric",
    detector: Detector,
    pipeline: "DetectionPipeline",
    with_dask: bool = False,
) -> "ParametricResult":
    """Run a 'parametric' pipeline.

    Parameters
    ----------
    parametric: Parametric
    detector: Detector
    pipeline: Pipeline
    with_dask: bool

    Returns
    -------
    result: ParametricResult
    """
    logging.info("Mode: Parametric")

    parametric_outputs = parametric.outputs  # type: ParametricOutputs
    detector.set_output_dir(parametric_outputs.output_dir)  # TODO: Remove this

    # TODO: This should be done during initializing of object `Configuration`
    # parametric_outputs.params_func(parametric)

    processor = Processor(detector=detector, pipeline=pipeline)

    result = parametric.run_parametric(processor=processor)

    if parametric_outputs.save_parametric_data:
        parametric_outputs.save_parametric_datasets(
            result=result, mode=parametric.parametric_mode
        )

    return result


def calibration_mode(
    calibration: "Calibration",
    detector: Detector,
    pipeline: "DetectionPipeline",
    compute_and_save: bool = True,
) -> t.Tuple[xr.Dataset, pd.DataFrame, pd.DataFrame, t.Sequence]:
    """Run a 'calibration' pipeline.

    Parameters
    ----------
    calibration: Calibration
    detector: Detector
    pipeline: DetectionPipeline
    compute_and_save: bool

    Returns
    -------
    tuple
    """
    logging.info("Mode: Calibration")

    calibration_outputs = calibration.outputs  # type: CalibrationOutputs
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
    filenames = calibration.post_processing(
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


def output_directory(configuration: Configuration) -> Path:
    """Return the output directory from the configuration.

    Parameters
    ----------
    configuration

    Returns
    -------
    output_dir
    """
    if isinstance(configuration.exposure, Exposure):
        output_dir = configuration.exposure.outputs.output_dir
    elif isinstance(configuration.calibration, Calibration):
        output_dir = configuration.calibration.outputs.output_dir
    elif isinstance(configuration.parametric, Parametric):
        output_dir = configuration.parametric.outputs.output_dir
    else:
        raise (ValueError("Outputs not initialized."))
    return output_dir


def run(input_filename: str, random_seed: t.Optional[int] = None) -> None:
    """TBW.

    Parameters
    ----------
    input_filename
    random_seed
    """
    logging.info("Pyxel version %s", version)
    logging.info("Pipeline started.")

    start_time = time.time()
    if random_seed:
        np.random.seed(random_seed)

    configuration = load(
        Path(input_filename).expanduser().resolve()
    )  # type: Configuration

    output_dir = output_directory(configuration)

    save(input_filename=input_filename, output_dir=output_dir)

    pipeline = configuration.pipeline  # type: DetectionPipeline

    if isinstance(configuration.ccd_detector, CCD):
        detector = configuration.ccd_detector  # type: t.Union[CCD, CMOS, MKID]
    elif isinstance(configuration.cmos_detector, CMOS):
        detector = configuration.cmos_detector
    elif isinstance(configuration.mkid_detector, MKID):
        detector = configuration.mkid_detector
    else:
        raise NotImplementedError("Detector is not defined in YAML config. file!")

    if isinstance(configuration.exposure, Exposure):
        exposure = configuration.exposure  # type: Exposure
        exposure_mode(exposure=exposure, detector=detector, pipeline=pipeline)

    elif isinstance(configuration.calibration, Calibration):

        calibration = configuration.calibration  # type: Calibration
        _ = calibration_mode(
            calibration=calibration, detector=detector, pipeline=pipeline
        )

    elif isinstance(configuration.parametric, Parametric):
        parametric = configuration.parametric  # type: Parametric
        parametric_mode(parametric=parametric, detector=detector, pipeline=pipeline)

    else:
        raise NotImplementedError("Please provide a valid simulation mode !")

    logging.info("Pipeline completed.")
    logging.info("Running time: %.3f seconds" % (time.time() - start_time))
    # Closing the logger in order to be able to move the file in the output dir
    logging.shutdown()
    outputs.save_log_file(output_dir)
    plt.close()


# TODO: Use library 'click' instead of 'parser' ? See issue #62
#       Add an option to display colors ? (very optional)
def main() -> None:
    """Define the argument parser and run Pyxel."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__
    )

    parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=0,
        help="Increase output verbosity (-v/-vv/-vvv)",
    )

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="Pyxel, version {version}".format(version=version),
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Configuration file to load (YAML)",
    )

    parser.add_argument("-s", "--seed", type=int, help="Random seed for the framework")

    parser.add_argument(
        "--download-examples",
        nargs="?",
        const="pyxel-examples",
        help="Install examples to the specified directory, default is /pyxel-examples.",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force flag for saving the examples.",
    )

    parser.add_argument(
        "-cm",
        "--createmodel",
        type=str,
        help="""Use: -cm arg1/arg2. Create a new module in\
        pyxel/models/arg1/arg2 using a template\
        (pyxel/templates/MODELTEMPLATE.py)""",
    )

    # parser.add_argument('-g', '--gui', default=False, type=bool, help='run Graphical User Interface')
    # parser.add_argument('-p', '--port', default=9999, type=int, help='The port to run the web server on')

    opts = parser.parse_args()

    logging_level = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG][
        min(opts.verbosity, 3)
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

    if opts.config:
        run(input_filename=opts.config, random_seed=opts.seed)
    elif opts.download_examples:
        download_examples(foldername=opts.download_examples, force=opts.force)
    elif opts.createmodel:
        create_model(newmodel=opts.createmodel)
    else:
        print("Define a YAML configuration file!")


if __name__ == "__main__":
    main()

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

import numpy as np
from matplotlib import pyplot as plt

from pyxel import __version__ as version
from pyxel import inputs_outputs as io
from pyxel.calibration import Calibration
from pyxel.detectors import CCD, CMOS
from pyxel.dynamic import Dynamic
from pyxel.inputs_outputs import Configuration
from pyxel.parametric import Parametric
from pyxel.pipelines import DetectionPipeline, Processor
from pyxel.single import Single
from pyxel.util import download_examples

if t.TYPE_CHECKING:
    from .inputs_outputs import (
        CalibrationOutputs,
        DynamicOutputs,
        ParametricOutputs,
        Result,
        SingleOutputs,
    )


def single_mode(
    single: "Single",
    detector: t.Union["CCD", "CMOS"],
    pipeline: "DetectionPipeline",
) -> None:
    """Run a 'single' pipeline.

    Parameters
    ----------
    single
    detector
    pipeline

    Returns
    -------
    None
    """
    logging.info("Mode: Single")

    single_outputs = single.outputs  # type: SingleOutputs
    detector.set_output_dir(single_outputs.output_dir)  # TODO: Remove this

    processor = Processor(detector=detector, pipeline=pipeline)

    _ = processor.run_pipeline()

    single_outputs.save_to_file(processor)
    single_outputs.single_to_plot(processor)


def parametric_mode(
    parametric: "Parametric",
    detector: t.Union["CCD", "CMOS"],
    pipeline: "DetectionPipeline",
    with_dask: bool = False,
) -> None:
    """Run a 'parametric' pipeline.

    Parameters
    ----------
    parametric
    detector
    pipeline
    with_dask

    Returns
    -------
    None
    """
    logging.info("Mode: Parametric")

    parametric_outputs = parametric.outputs  # type: ParametricOutputs
    detector.set_output_dir(parametric_outputs.output_dir)  # TODO: Remove this

    # TODO: This should be done during initializing of object `Configuration`
    # parametric_outputs.params_func(parametric)

    processor = Processor(detector=detector, pipeline=pipeline)

    parametric.run_parametric(processor=processor)


def dynamic_mode(
    dynamic: "Dynamic",
    detector: t.Union["CCD", "CMOS"],
    pipeline: "DetectionPipeline",
) -> None:
    """Run a 'dynamic' pipeline.

    Parameters
    ----------
    dynamic
    detector
    pipeline

    Returns
    -------
    None
    """

    logging.info("Mode: Dynamic")

    dynamic_outputs = dynamic.outputs  # type: DynamicOutputs

    detector.set_output_dir(dynamic_outputs.output_dir)  # TODO: Remove this

    processor = Processor(detector=detector, pipeline=pipeline)

    if isinstance(detector, CCD):
        dynamic.non_destructive_readout = False

    detector.set_dynamic(
        steps=dynamic.steps,
        time_step=dynamic.t_step,
        ndreadout=dynamic.non_destructive_readout,
    )

    # TODO: Use an iterator for that ?
    while detector.elapse_time():
        logging.info("time = %.3f s", detector.time)
        if detector.is_non_destructive_readout:
            detector.initialize(reset_all=False)
        else:
            detector.initialize(reset_all=True)
        processor.run_pipeline()
        if detector.read_out:
            dynamic_outputs.single_output(processor)


def calibration_mode(
    calibration: "Calibration",
    detector: t.Union["CCD", "CMOS"],
    pipeline: "DetectionPipeline",
) -> t.Tuple:
    """Run a 'calibration' pipeline.

    Parameters
    ----------
    calibration
    detector
    pipeline

    Returns
    -------
    None
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

    return ds_results, df_processors, df_all_logs, filenames


def output_directory(configuration: Configuration) -> Path:
    """Return the output directory from the configuration.

    Parameters
    ----------
    configuration

    Returns
    -------
    output_dir
    """
    if isinstance(configuration.single, Single):
        output_dir = configuration.single.outputs.output_dir
    elif isinstance(configuration.calibration, Calibration):
        output_dir = configuration.calibration.outputs.output_dir
    elif isinstance(configuration.dynamic, Dynamic):
        output_dir = configuration.dynamic.outputs.output_dir
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

    configuration = io.load(
        Path(input_filename).expanduser().resolve()
    )  # type: Configuration

    output_dir = output_directory(configuration)

    io.save(input_filename=input_filename, output_dir=output_dir)

    pipeline = configuration.pipeline  # type: DetectionPipeline

    if isinstance(configuration.ccd_detector, CCD):
        detector = configuration.ccd_detector  # type: t.Union[CCD, CMOS]
    elif isinstance(configuration.cmos_detector, CMOS):
        detector = configuration.cmos_detector
    else:
        raise NotImplementedError("Detector is not defined in YAML config. file!")

    if isinstance(configuration.single, Single):
        single = configuration.single  # type: Single
        single_mode(single=single, detector=detector, pipeline=pipeline)

    elif isinstance(configuration.calibration, Calibration):

        calibration = configuration.calibration  # type: Calibration
        _ = calibration_mode(
            calibration=calibration, detector=detector, pipeline=pipeline
        )

    elif isinstance(configuration.parametric, Parametric):
        parametric = configuration.parametric  # type: Parametric
        parametric_mode(parametric=parametric, detector=detector, pipeline=pipeline)

    elif isinstance(configuration.dynamic, Dynamic):

        dynamic = configuration.dynamic  # type: Dynamic
        dynamic_mode(dynamic=dynamic, detector=detector, pipeline=pipeline)

    else:
        raise NotImplementedError("Please provide a valid simulation mode !")

    logging.info("Pipeline completed.")
    logging.info("Running time: %.3f seconds" % (time.time() - start_time))
    # Closing the logger in order to be able to move the file in the output dir
    logging.shutdown()
    io.save_log_file(output_dir)
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
    else:
        print("Define a YAML configuration file!")


if __name__ == "__main__":
    main()

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
from dask import delayed
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

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


def single_mode(processor: Processor, single: "Single") -> None:
    """Run a 'single' pipeline.

    Parameters
    ----------
    processor
    single

    Returns
    -------
    None
    """
    logging.info("Mode: Single")

    single_outputs = single.outputs  # type: SingleOutputs
    processor.detector.set_output_dir(single_outputs.output_dir)  # TODO: Remove this

    _ = processor.run_pipeline()

    single_outputs.save_to_file(processor)
    single_outputs.single_to_plot(processor)


def parametric_mode(
    processor: Processor,
    parametric: Parametric,
    with_dask: bool = False,
) -> None:
    """Run a 'parametric' pipeline.

    Parameters
    ----------
    processor
    parametric
    with_dask

    Returns
    -------
    None
    """
    logging.info("Mode: Parametric")

    parametric_outputs = parametric.outputs  # type: ParametricOutputs
    processor.detector.set_output_dir(
        parametric_outputs.output_dir
    )  # TODO: Remove this

    # TODO: This should be done during initializing of object `Configuration`
    # parametric_outputs.params_func(parametric)

    # Check if all keys from 'parametric' are valid keys for object 'pipeline'
    for param_value in parametric.enabled_steps:
        key = param_value.key  # type: str
        assert processor.has(key)

    processors_it = parametric.collect(processor)  # type: t.Iterator[Processor]

    result_list = []  # type: t.List[Result]
    output_filenames = []  # type: t.List[t.Sequence[Path]]

    # Run all pipelines
    for proc in tqdm(processors_it):  # type: Processor

        if not with_dask:
            result_proc = proc.run_pipeline()  # type: Processor
            result_val = parametric_outputs.extract_func(
                processor=result_proc
            )  # type: Result

            filenames = parametric_outputs.save_to_file(
                processor=result_proc
            )  # type: t.Sequence[Path]

        else:
            result_proc = delayed(proc.run_pipeline)()
            result_val = delayed(parametric_outputs.extract_func)(processor=result_proc)

            filenames = delayed(parametric_outputs.save_to_file)(processor=result_proc)

        result_list.append(result_val)
        output_filenames.append(filenames)  # TODO: This is not used

    if not with_dask:
        plot_array = parametric_outputs.merge_func(result_list)  # type: np.ndarray
    else:
        array = delayed(parametric_outputs.merge_func)(result_list)
        plot_array, _ = dask.compute(array, output_filenames)

    # TODO: Plot with dask ?
    if parametric_outputs.parametric_plot is not None:
        parametric_outputs.plotting_func(plot_array)


def dynamic_mode(processor: "Processor", dynamic: "Dynamic") -> None:
    """Run a 'dynamic' pipeline.

    Parameters
    ----------
    processor
    dynamic

    Returns
    -------
    None
    """

    logging.info("Mode: Dynamic")

    dynamic_outputs = dynamic.outputs  # type: DynamicOutputs

    detector = processor.detector
    detector.set_output_dir(dynamic_outputs.output_dir)  # TODO: Remove this

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


def calibration_mode(processor: "Processor", calibration: "Calibration") -> t.Tuple:
    """Run a 'calibration' pipeline.

    Parameters
    ----------
    processor
    calibration

    Returns
    -------
    None
    """
    logging.info("Mode: Calibration")

    calibration_outputs = calibration.outputs  # type: CalibrationOutputs
    processor.detector.set_output_dir(
        calibration_outputs.output_dir
    )  # TODO: Remove this

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

    processor = Processor(detector=detector, pipeline=pipeline)

    if isinstance(configuration.single, Single):
        single = configuration.single  # type: Single
        single_mode(processor=processor, single=single)

    elif isinstance(configuration.calibration, Calibration):

        calibration = configuration.calibration  # type: Calibration
        _ = calibration_mode(processor=processor, calibration=calibration)

    elif isinstance(configuration.parametric, Parametric):
        parametric = configuration.parametric  # type: Parametric
        parametric_mode(processor=processor, parametric=parametric)

    elif isinstance(configuration.dynamic, Dynamic):

        dynamic = configuration.dynamic  # type: Dynamic
        dynamic_mode(processor=processor, dynamic=dynamic)

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

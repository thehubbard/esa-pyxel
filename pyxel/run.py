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
from pyxel.util import Outputs
from pyxel.util.outputs import Result


def single_mode(processor: Processor, out: Outputs) -> plt.Figure:
    """TBW.

    Parameters
    ----------
    processor
    out

    Returns
    -------
    Figure
        TBW.
    """
    logging.info("Mode: Single")

    _ = processor.run_pipeline()

    out.save_to_file(processor)
    out.single_to_plot(processor)

    return out.fig


def parametric_mode(
    processor: Processor,
    parametric: Parametric,
    output: Outputs,
    with_dask: bool = False,
) -> t.Optional[plt.Figure]:
    """Run a 'parametric' pipeline.

    Parameters
    ----------
    processor
    parametric
    output
    with_dask

    Returns
    -------
    Optional `Figure`
        TBW.
    """
    logging.info("Mode: Parametric")

    # Check if all keys from 'parametric' are valid keys for object 'pipeline'
    for param_value in parametric.enabled_steps:
        key = param_value.key  # type: str
        assert processor.has(key)

    processors_it = parametric.collect(processor)  # type: t.Iterator[Processor]

    result_list = []  # type: t.List[Result]
    output_filenames = []  # type: t.List[t.Sequence[Path]]

    # out.params_func(parametric)

    # Run all pipelines
    for proc in tqdm(processors_it):  # type: Processor

        if not with_dask:
            result_proc = proc.run_pipeline()  # type: Processor
            result_val = output.extract_func(processor=result_proc)  # type: Result

            filenames = output.save_to_file(
                processor=result_proc
            )  # type: t.Sequence[Path]

        else:
            result_proc = delayed(proc.run_pipeline)()
            result_val = delayed(output.extract_func)(processor=result_proc)

            filenames = delayed(output.save_to_file)(processor=result_proc)

        result_list.append(result_val)
        output_filenames.append(filenames)  # TODO: This is not used

    if not with_dask:
        plot_array = output.merge_func(result_list)  # type: np.ndarray
    else:
        array = delayed(output.merge_func)(result_list)
        plot_array, _ = dask.compute(array, output_filenames)

    # TODO: Plot with dask ?
    fig = None  # type: t.Optional[plt.Figure]
    if output.parametric_plot is not None:
        output.plotting_func(plot_array)
        fig = output.fig

    return fig


def dynamic_mode(processor: "Processor", dynamic: "Dynamic", outputs: "Outputs"):

    logging.info("Mode: Dynamic")

    detector = processor.detector

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
            outputs.single_output(processor)


def calibration_mode(
    processor: "Processor", calibration: "Calibration", outputs: "Outputs"
):

    logging.info("Mode: Calibration")

    results = calibration.run_calibration(
        processor=processor, output_dir=outputs.output_dir
    )

    calibration.post_processing(calib_results=results, output=outputs)


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

    pipeline = configuration.pipeline  # type: DetectionPipeline

    if hasattr(configuration, "ccd_detector"):
        detector = configuration.ccd_detector  # type: CCD
    elif hasattr(configuration, "cmos_detector"):
        detector = configuration.cmos_detector  # type: CMOS
    else:
        raise NotImplementedError("Detector is not defined in YAML config. file!")

    processor = Processor(detector=detector, pipeline=pipeline)

    if hasattr(configuration, "single"):

        single = configuration.single  # type: Single

        outputs = single.outputs  # type: Outputs
        outputs.copy_config_file(input_filename)
        detector.set_output_dir(outputs.output_dir)  # TODO: Remove this

        _ = single_mode(processor=processor, out=outputs)

    elif hasattr(configuration, "calibration"):

        calibration = configuration.calibration  # type: Calibration

        outputs = calibration.outputs  # type: Outputs
        outputs.copy_config_file(input_filename)
        detector.set_output_dir(outputs.output_dir)  # TODO: Remove this

        calibration_mode(processor=processor, calibration=calibration, outputs=outputs)

    elif hasattr(configuration, "parametric"):

        parametric = configuration.parametric  # type: Parametric

        outputs = parametric.outputs  # type: Outputs
        outputs.copy_config_file(input_filename)
        detector.set_output_dir(outputs.output_dir)  # TODO: Remove this

        # TODO: This should be done during initializing of object `Configuration`
        # out.params_func(parametric)

        _ = parametric_mode(processor=processor, parametric=parametric, output=outputs)

    elif hasattr(configuration, "dynamic"):

        dynamic = configuration.dynamic  # type: Dynamic

        outputs = dynamic.outputs  # type: Outputs
        outputs.copy_config_file(input_filename)
        detector.set_output_dir(outputs.output_dir)  # TODO: Remove this

        dynamic_mode(processor=processor, dynamic=dynamic, output=outputs)

    else:
        raise NotImplementedError(f"Please provide a valid simulation mode !")

    logging.info("Pipeline completed.")
    logging.info("Running time: %.3f seconds" % (time.time() - start_time))
    # Closing the logger in order to be able to move the file in the output dir
    logging.shutdown()
    outputs.save_log_file()
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
        required=True,
        help="Configuration file to load (YAML)",
    )
    parser.add_argument("-s", "--seed", type=int, help="Random seed for the framework")

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
    else:
        print("Define a YAML configuration file!")


if __name__ == "__main__":
    main()

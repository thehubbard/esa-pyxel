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
import pyxel.io as io
from dask import delayed, distributed
from dask.delayed import Delayed
from pyxel import __version__ as version
from pyxel.detectors import CCD, CMOS
from pyxel.parametric.parametric import Configuration
from pyxel.pipelines.pipeline import DetectionPipeline
from pyxel.pipelines.processor import Processor
from pyxel.util import Outputs


def run(input_filename: str, random_seed: t.Optional[int] = None) -> None:
    """TBW.

    :param input_filename:
    :param random_seed:
    :return:
    """
    logging.info("Pyxel version %s", version)
    logging.info("Pipeline started.")

    start_time = time.time()
    if random_seed:
        np.random.seed(random_seed)

    # TODO: 'cfg' is a `dict`. It would better to use an helper class. See Issue #60.
    #       Example:
    #           >>> cfg = io.load(input_filename)
    #           >>> cfg.pipeline
    #           ...
    #           >>> cfg.simulation
    #           ...
    cfg = io.load(Path(input_filename))  # type: dict

    pipeline = cfg["pipeline"]  # type: DetectionPipeline

    simulation = cfg["simulation"]  # type: Configuration

    if "ccd_detector" in cfg:
        detector = cfg["ccd_detector"]  # type: t.Union[CCD, CMOS]
    elif "cmos_detector" in cfg:
        detector = cfg["cmos_detector"]
    else:
        raise NotImplementedError("Detector is not defined in YAML config. file!")

    processor = Processor(detector=detector, pipeline=pipeline)

    out = simulation.outputs  # type: Outputs
    out.set_input_file(input_filename)
    detector.set_output_dir(out.output_dir)

    # TODO: Create new separate functions 'run_single', 'run_calibration', 'run_parametric'
    #       and 'run_dynamic'. See issue #61.
    if simulation.mode == "single":
        logging.info("Mode: Single")

        processor.run_pipeline()
        out.single_output(processor)

    elif simulation.mode == "calibration":
        if not simulation.calibration:
            raise RuntimeError("Missing 'Calibration' parameters.")

        logging.info("Mode: Calibration")
        results = simulation.calibration.run_calibration(
            processor=processor, output_dir=out.output_dir
        )

        simulation.calibration.post_processing(calib_results=results, output=out)

    elif simulation.mode == "parametric":
        if not simulation.parametric:
            raise RuntimeError("Missing 'Parametric' parameters.")

        logging.info("Mode: Parametric")

        # client = distributed.Client(processes=True)
        # client = distributed.Client(n_workers=4, processes=False, threads_per_worker=4)
        client = distributed.Client(processes=False)
        logging.info(client)
        # use as few processes (and workers?) as possible with as many threads_per_worker as possible
        # Dasbboard available on http://127.0.0.1:8787

        configs = simulation.parametric.collect(processor)
        result_list = []
        out.params_func(simulation.parametric)
        for proc in configs:
            result_proc = delayed(proc.run_pipeline)()
            result_val = delayed(out.extract_func)(proc=result_proc)
            result_list.append(result_val)
        array = delayed(out.merge_func)(result_list)  # type: Delayed
        plot_array = array.compute()
        if out.parametric_plot is not None:
            out.plotting_func(plot_array)

    elif simulation.mode == "dynamic":
        if not simulation.dynamic:
            raise RuntimeError("Missing 'Dynamic' parameters.")

        logging.info("Mode: Dynamic")
        if "non_destructive_readout" not in simulation.dynamic or isinstance(
            detector, CCD
        ):
            simulation.dynamic["non_destructive_readout"] = False
        if "t_step" in simulation.dynamic and "steps" in simulation.dynamic:
            detector.set_dynamic(
                steps=simulation.dynamic["steps"],
                time_step=simulation.dynamic["t_step"],
                ndreadout=simulation.dynamic["non_destructive_readout"],
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
                out.single_output(processor)

    else:
        raise NotImplementedError(
            f"Simulation mode {simulation.mode} is not implemented !"
        )

    logging.info("Pipeline completed.")
    logging.info("Running time: %.3f seconds" % (time.time() - start_time))
    # Closing the logger in order to be able to move the file in the output dir
    logging.shutdown()
    out.save_log_file()


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

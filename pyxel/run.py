#   --------------------------------------------------------------------------
#   Copyright 2019 SCI-FIV, ESA (European Space Agency)
#   --------------------------------------------------------------------------
"""Pyxel detector simulation framework.

Pyxel is a detector simulation framework, that can simulate a variety of
detector effects (e.g., cosmic rays, radiation-induced CTI in CCDs, persistence
in MCT, charge diffusion, crosshatches, noises, crosstalk etc.) on a given image.
"""
import argparse
import logging
import sys
import time
from pathlib import Path
import numpy as np
import pyxel.io as io
from pyxel.pipelines.processor import Processor
from pyxel.detectors import CCD, CMOS
from pyxel import __version__ as version
import typing as t
from pyxel.pipelines.pipeline import DetectionPipeline


def run(input_filename: str, random_seed: t.Optional[int] = None) -> None:
    """TBW.

    :param input_filename:
    :param random_seed:
    :return:
    """
    logging.info('Pyxel version ' + version)
    logging.info('Pipeline started.')

    start_time = time.time()
    if random_seed:
        np.random.seed(random_seed)

    # FRED: 'cfg' is a `dict`. It would better to use an object create from a class
    #       built by 'esapy_config'
    cfg = io.load(Path(input_filename))

    pipeline = cfg['pipeline']  # type: DetectionPipeline

    simulation = cfg['simulation']

    if 'ccd_detector' in cfg:
        detector = cfg['ccd_detector']  # type: t.Union[CCD, CMOS]
    elif 'cmos_detector' in cfg:
        detector = cfg['cmos_detector']
    else:
        raise NotImplementedError
        # detector = cfg['ccd_detector']  # type: CCD

    processor = Processor(detector, pipeline)

    out = simulation.outputs
    if out:
        out.set_input_file(input_filename)
        detector.set_output_dir(out.output_dir)

    # HANS: place all code in each if block into a separate function
    #   and use a dict call map. Example:
    #   mode_funcs = {
    #       'single': run_single_mode,
    #       'calibration': run_calibration_mode,
    #       'parametric': run_parametric_mode,
    #       ... etc ...
    #   }

    # HANS: place logger Mode line outside if / elif /else block. Example:
    #   logger.info('Mode: %r', simulation.mode)
    if simulation.mode == 'single':
        logging.info('Mode: Single')
        processor.pipeline.run_pipeline(detector)
        if out:
            out.single_output(processor)

    elif simulation.mode == 'calibration' and simulation.calibration:
        logging.info('Mode: Calibration')
        processor, results = simulation.calibration.run_calibration(processor, out)
        logging.info('Champion fitness:   %1.5e' % results['fitness'])
        if out:
            out.calibration_output(processor=processor, results=results)

    elif simulation.mode == 'parametric' and simulation.parametric:
        logging.info('Mode: Parametric')
        configs = simulation.parametric.collect(processor)
        for proc in configs:
            proc.pipeline.run_pipeline(proc.detector)
            if out:
                out.add_parametric_step(processor=proc,
                                        parametric=simulation.parametric)
        if out:
            out.parametric_output()

    elif simulation.mode == 'dynamic' and simulation.dynamic:
        logging.info('Mode: Dynamic')
        if 'non_destructive_readout' not in simulation.dynamic or isinstance(detector, CCD):
            simulation.dynamic['non_destructive_readout'] = False
        if 't_step' in simulation.dynamic and 'steps' in simulation.dynamic:
            detector.set_dynamic(steps=simulation.dynamic['steps'],
                                 time_step=simulation.dynamic['t_step'],
                                 ndreadout=simulation.dynamic['non_destructive_readout'])
        while detector.elapse_time():  # FRED: Use an iterator for that ?
            logging.info('time = %.3f s' % detector.time)
            if detector.is_non_destructive_readout:
                detector.initialize(reset_all=False)
            else:
                detector.initialize(reset_all=True)
            processor.pipeline.run_pipeline(detector)
            if out and detector.read_out:
                out.single_output(processor)

    else:
        raise ValueError

    logging.info('Pipeline completed.')
    logging.info('Running time: %.3f seconds' % (time.time() - start_time))
    # Closing the logger in order to be able to move the file in the output dir
    logging.shutdown()
    if out:
        out.save_log_file()


# FRED: Add an option to display colors ? (very optional)
# FRED: Use library 'click' instead of 'parser' ? (very optional)
def main():
    """Define the argument parser and run Pyxel."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description=__doc__)

    parser.add_argument('-v', '--verbosity', action='count', default=0, help='Increase output verbosity (-v/-vv/-vvv)')
    parser.add_argument('-V', '--version', action='version',
                        version='Pyxel, version {version}'.format(version=version))
    parser.add_argument('-c', '--config', type=str, required=True, help='Configuration file to load (YAML)')
    parser.add_argument('-s', '--seed', type=int, help='Random seed for the framework')

    # parser.add_argument('-g', '--gui', default=False, type=bool, help='run Graphical User Interface')
    # parser.add_argument('-p', '--port', default=9999, type=int, help='The port to run the web server on')

    opts = parser.parse_args()

    logging_level = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG][min(opts.verbosity, 3)]
    log_format = '%(asctime)s - %(name)s - %(funcName)30s \t %(message)s'
    logging.basicConfig(filename='pyxel.log',
                        level=logging_level,
                        format=log_format,
                        datefmt='%d-%m-%Y %H:%M:%S')
    # If user wants the log in stdout AND in file, use the three lines below
    stream_stdout = logging.StreamHandler(sys.stdout)
    stream_stdout.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(stream_stdout)

    if opts.config:
        run(input_filename=opts.config, random_seed=opts.seed)
    else:
        print('Define a YAML configuration file!')


if __name__ == '__main__':
    main()

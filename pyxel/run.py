#   --------------------------------------------------------------------------
#   Copyright 2018 SCI-FIV, ESA (European Space Agency)
#   --------------------------------------------------------------------------
"""Pyxel detector simulation framework.

Pyxel is a detector simulation framework, that can simulate a variety of
detector effects (e.g., cosmics, radiation-induced CTI  in CCDs, persistency
in MCT, diffusion, cross-talk etc.) on a given image.
"""
import argparse
import logging
import time
from pathlib import Path
import numpy as np
import esapy_config.io as io
import pyxel
from pyxel.pipelines.processor import Processor
from pyxel.detectors.cmos import CMOS


def run(input_filename, random_seed: int = None):
    """TBW.

    :param input_filename:
    :param random_seed:
    :return:
    """
    logger = logging.getLogger('pyxel')
    logger.info('Pipeline started.')
    start_time = time.time()
    if random_seed:
        np.random.seed(random_seed)

    cfg = io.load(Path(input_filename))
    simulation = cfg['simulation']
    processor = Processor(cfg['detector'],
                          cfg['pipeline'])
    out = simulation.outputs
    if out:
        out.set_input_file(input_filename)
    else:
        logger.warning('Output is not defined! No output files will be saved!')
    processor.detector.set_output_dir(out.output_dir)

    if simulation.mode == 'single':
        logger.info('Mode: Single')
        processor.pipeline.run_pipeline(processor.detector)
        if out:
            out.single_output(processor)

    elif simulation.mode == 'calibration' and simulation.calibration:
        logger.info('Mode: Calibration')
        processor, results = simulation.calibration.run_calibration(processor, out)
        logger.info('Champion fitness:   %1.5e' % results['fitness'])
        if out:
            out.calibration_output(processor=processor, results=results)

    elif simulation.mode == 'parametric' and simulation.parametric:
        logger.info('Mode: Parametric')
        configs = simulation.parametric.collect(processor)
        for processor in configs:
            processor.pipeline.run_pipeline(processor.detector)
            if out:
                out.add_parametric_step(processor=processor,
                                        parametric=simulation.parametric)
        if out:
            out.parametric_output()

    elif simulation.mode == 'dynamic' and simulation.dynamic:
        if not isinstance(processor.detector, CMOS):
            raise TypeError('Dynamic mode only works with CMOS detector!')
        logger.info('Mode: Dynamic')

        if 't_start' not in simulation.dynamic:
            simulation.dynamic['t_start'] = 0.
        if 'non_destructive_readout' not in simulation.dynamic:
            simulation.dynamic['non_destructive_readout'] = False

        processor.detector.set_dynamic(start_time=simulation.dynamic['t_start'],
                                       end_time=simulation.dynamic['t_end'],
                                       time_steps=simulation.dynamic['t_steps'],
                                       ndreadout=simulation.dynamic['non_destructive_readout'])
        for processor.detector.time in np.linspace(processor.detector.start_time, processor.detector.end_time,
                                                   num=processor.detector.time_steps, endpoint=True):
            logger.info('time = %.2e s' % processor.detector.time)
            if not processor.detector.is_non_destructive_readout:
                processor.detector.initialize()
            processor.pipeline.run_pipeline(processor.detector)
            if out:
                out.single_output(processor)
        pass

    else:
        raise ValueError

    logger.info('Pipeline completed.')
    logger.info('Running time: %.3f seconds' % (time.time() - start_time))


def main():
    """Define the argument parser and run Pyxel."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description=__doc__)

    parser.add_argument('-v', '--verbosity', action='count', default=0, help='Increase output verbosity (-v/-vv/-vvv)')
    parser.add_argument('-V', '--version', action='version',
                        version='Pyxel, version {version}'.format(version=pyxel.__version__))
    parser.add_argument('-c', '--config', type=str, required=True, help='Configuration file to load (YAML)')
    parser.add_argument('-s', '--seed', type=int, help='Random seed for the framework')

    # parser.add_argument('-o', '--output', type=str, default='outputs', help='Path for output folder')
    # parser.add_argument('-g', '--gui', default=False, type=bool, help='run Graphical User Interface')
    # parser.add_argument('-p', '--port', default=9999, type=int, help='The port to run the web server on')

    opts = parser.parse_args()

    logging_level = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG][min(opts.verbosity, 3)]
    log_format = '%(asctime)s - %(name)s - %(funcName)20s \t %(message)s'
    logging.basicConfig(level=logging_level, format=log_format, datefmt='%d-%m-%Y %H:%M:%S')

    if opts.config:
        run(input_filename=opts.config, random_seed=opts.seed)   # output_directory=opts.output,
    else:
        print('Define a YAML configuration file!')


if __name__ == '__main__':
    main()

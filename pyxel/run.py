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
import os
from shutil import copy2
from pathlib import Path
import numpy as np
import esapy_config.io as io
import pyxel
from pyxel.pipelines.processor import Processor
from pyxel.util import Outputs, apply_run_number


def single_output(detector, output_dir):
    """TBW."""
    out = Outputs(output_dir)
    out.save_to_fits(array=detector.image.array)
    out.save_to_npy(array=detector.image.array)
    plt_args = {'bins': 300, 'xlabel': 'ADU', 'ylabel': 'counts', 'title': 'Image histogram'}
    out.plot_histogram(detector.image.array, name='image', arg_dict=plt_args)
    out.save_plot()
    plt_args = {'axis': [3000, 6000, 3000, 6000]}
    out.plot_graph(detector.image.array, detector.image.array, name='image', arg_dict=plt_args)
    out.save_plot()
    # out.save_to_bitmap(array=detector.image.array)
    # out.save_to_hdf(data=detector.charges.frame, key='charge')
    # out.save_to_csv(dataframe=detector.charges.frame)


def calibration_output(results):        # TODO
    """TBW."""
    pass


def parametric_output(detector, output_dir, config=None):        # TODO
    """TBW."""
    pass
    # out = Outputs(output_dir)
    # out.save_to_fits(array=detector.image.array)
    # out.plot_histogram(detector.image.array, name='image_hist')
    # out.save_plot('graph')
    # todo: get the parametric variables from configs,
    # todo: then plot things in function of these variables, defined in configs


def run(input_filename, output_directory: str, random_seed: int = None):
    """TBW.

    :param input_filename:
    :param output_directory:
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
    detector = cfg['detector']
    pipeline = cfg['pipeline']
    processor = Processor(detector, pipeline)

    output_directory = apply_run_number(output_directory + '/run_??')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    else:
        raise IsADirectoryError('Directory exists.')
    copy2(input_filename, output_directory)

    if simulation.mode == 'single':
        logger.info('Mode: Single')
        detector = processor.pipeline.run_pipeline(processor.detector)
        single_output(detector=detector, output_dir=output_directory)

    elif simulation.mode == 'calibration':
        logger.info('Mode: Calibration')
        calibration_results = simulation.calibration.run_calibration(processor)
        # TODO: return the optimal model/detector parameters as dict or obj
        # calibration_output(calibration_results)

    elif simulation.mode == 'parametric':
        logger.info('Mode: Parametric')
        configs = simulation.parametric.collect(processor)
        for config in configs:
            detector = config.pipeline.run_pipeline(config.detector)
            parametric_output(detector=detector, output_dir=output_directory)

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
    parser.add_argument('-o', '--output', type=str, default='outputs', help='Path for output folder')
    parser.add_argument('-s', '--seed', type=int, help='Random seed for the framework')

    # parser.add_argument('-g', '--gui', default=False, type=bool, help='run Graphical User Interface')
    # parser.add_argument('-p', '--port', default=9999, type=int, help='The port to run the web server on')

    opts = parser.parse_args()

    logging_level = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG][min(opts.verbosity, 3)]
    log_format = '%(asctime)s - %(name)s - %(funcName)20s \t %(message)s'
    logging.basicConfig(level=logging_level, format=log_format, datefmt='%d-%m-%Y %H:%M:%S')

    if opts.config:
        run(input_filename=opts.config, output_directory=opts.output, random_seed=opts.seed)
    else:
        print('Define a YAML configuration file!')


if __name__ == '__main__':
    main()

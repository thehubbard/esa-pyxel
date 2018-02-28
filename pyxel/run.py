#   --------------------------------------------------------------------------
#   Copyright 2017 SRE-F, ESA (European Space Agency)
#   --------------------------------------------------------------------------
"""PYXEL detector simulation framework.

PYXEL is a detector simulation framework, that can simulate a variety of
detector effects (e.g., cosmics, radiation-induced CTI  in CCDs, persistency
in MCT, diffusion, cross-talk etc.) on a given image.
"""
import logging
import argparse
import typing as t   # noqa: F401
from pathlib import Path

import pyxel
from pyxel import util
import pyxel.pipelines.processor
from pyxel.io.yaml_processor import load_config


def run_parametric(input_filename, output_file):
    """TBW.

    :param input_filename:
    :param output_file:
    :return:
    """
    cfg = load_config(Path(input_filename))
    parametric = cfg.pop('parametric')
    processor = cfg[next(iter(cfg))]  # type: pyxel.pipelines.processor.Processor
    parametric.debug(processor)
    configs = parametric.collect(processor)
    for config in configs:
        detector = config.pipeline.run_pipeline(config.detector)

        if output_file:
            save_to = util.apply_run_number(output_file)
            out = util.FitsFile(save_to)
            out.save(detector.signal, header=None, overwrite=True)


def run_pipeline(input_filename, output_file):
    """TBW.

    :param input_filename:
    :param output_file:
    :return:
    """
    cfg = load_config(Path(input_filename))

    processor = cfg[next(iter(cfg))]  # type: pyxel.pipelines.processor.Processor

    pipeline = processor.pipeline  # type: t.Union[CCDDetectionPipeline, CMOSDetectionPipeline]

    # Run the pipeline
    detector = pipeline.run_pipeline(processor.detector)  # type: t.Union[CCD, CMOS]

    print('Pipeline completed.')

    if output_file:
        out = util.FitsFile(output_file)
        out.save(detector.signal, header=None, overwrite=True)      # TODO should replace result.signal to result.image


def main():
    """Define the argument parser and run the pipeline."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description=__doc__)

    parser.add_argument('-v', '--verbosity', action='count', default=0,
                        help='Increase output verbosity')
    parser.add_argument('--version', action='version',
                        version='%(prog)s (version {version})'.format(version=pyxel.__version__))

    # Required positional arguments
    parser.add_argument('-c', '--config',
                        help='Configuration file to load (YAML or INI)')

    # Required positional arguments
    parser.add_argument('-o', '--output',
                        help='output file')

    opts = parser.parse_args()

    # Set logger
    log_level = [logging.ERROR, logging.INFO, logging.DEBUG][min(opts.verbosity, 2)]
    log_format = '%(asctime)s - %(name)s - %(funcName)s - %(thread)d - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format)

    run_parametric(opts.config, opts.output)


if __name__ == '__main__':
    main()

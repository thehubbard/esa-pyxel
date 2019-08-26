"""Fix pattern noise model."""
import logging
from pathlib import Path
import numpy as np
from pyxel.detectors.detector import Detector
from pyxel.detectors.geometry import Geometry  # noqa: F401
import typing as t
# from astropy import units as u


# TODO: Fix this
# @validators.validate
# @config.argument(name='', label='', units='', validate=)
def fix_pattern_noise(detector: Detector,
                      pixel_non_uniformity: t.Optional[np.ndarray] = None) -> None:
    """Add fix pattern noise caused by pixel non-uniformity during charge collection.

    :param detector: Pyxel Detector object
    :param pixel_non_uniformity: path to an ascii file with and array
    """
    logging.info('')
    geo = detector.geometry  # type: Geometry

    if pixel_non_uniformity is not None:
        if '.npy' in pixel_non_uniformity:
            pnu = np.load(pixel_non_uniformity)
        else:
            pnu = np.loadtxt(pixel_non_uniformity)
    else:
        filename = 'data/pixel_non_uniformity.npy'
        if Path(filename).exists():
            logging.warning('"pixel_non_uniformity" file is not defined, '
                            'using array from file: ' + filename)
            pnu = np.load(filename)
        else:
            logging.warning('"pixel_non_uniformity" file is not defined, '
                            'generated random array to file: ' + filename)
            # pnu = 0.99 + np.random.random((geo.row, geo.col)) * 0.02
            pnu = np.random.normal(loc=1.0, scale=0.03, size=(geo.row, geo.col))
            np.save(filename, pnu)

    pnu = pnu.reshape((geo.row, geo.col))

    detector.pixel.array = detector.pixel.array.astype(np.float64) * pnu      # TODO: dtype!

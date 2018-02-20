#   --------------------------------------------------------------------------
#   Copyright 2018 SCI-FIV, ESA (European Space Agency)
#   --------------------------------------------------------------------------
"""PyXel! photon generator functions."""
import numpy as np
from astropy.io import fits
# from astropy import units as u

from pyxel.detectors.detector import Detector


def load_image(detector: Detector, image_file: str) -> Detector:
    """TBW.

    :param detector:
    :param image_file:
    :return:
    """
    geo = detector.geometry
    cht = detector.characteristics
    image = fits.getdata(image_file)
    geo.row, geo.col = image.shape
    photon_number_list = image / (cht.qe * cht.eta * cht.sv * cht.amp * cht.a1 * cht.a2)
    photon_number_list = np.rint(photon_number_list).astype(int).flatten()
    photon_energy_list = [0.] * geo.row * geo.col
    detector.photons.generate_photons(photon_number_list, photon_energy_list)

    return detector


def add_photon_level(detector: Detector, level: int) -> Detector:
    """TBW.

    :param detector:
    :param level:
    :return:
    """
    if level > 0:
        geo = detector.geometry
        photon_number_list = np.ones(geo.row * geo.col, dtype=int) * level
        photon_energy_list = [0.] * geo.row * geo.col
        detector.photons.generate_photons(photon_number_list, photon_energy_list)

    return detector

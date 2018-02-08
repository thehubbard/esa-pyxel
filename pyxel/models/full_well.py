#   --------------------------------------------------------------------------
#   Copyright 2018 SCI-FIV, ESA (European Space Agency)
#   --------------------------------------------------------------------------
"""
PyXel! CCD full well models
"""
import copy
import numpy as np

from pyxel.detectors.ccd import CCD


def simple_pixel_full_well(detector: CCD,
                           fwc: int = None) -> CCD:
    """
    Simply removing charges from pixels due to full well
    :return:
    """

    new_detector = copy.deepcopy(detector)

    charge = new_detector.pixels.get_pixel_charges()

    excess_pos = np.where(charge > fwc)
    charge[excess_pos] = fwc

    charge = np.rint(charge).astype(int)
    new_detector.pixels.change_all_charges(charge)
    # TODO add np.rint and np.int to Pixels class funcs

    return new_detector


def mc_full_well(detector: CCD,
                 fwc: np.ndarray = None) -> CCD:
    """
    Moving charges to random neighbour pixels due to full well which depends on pixel location
    :return:
    """

    new_detector = copy.deepcopy(detector)

    # detector.charges

    # pix_rows = new_detector.pixels.get_pixel_positions_ver()
    # pix_cols = new_detector.pixels.get_pixel_positions_hor()
    #
    # charge = np.zeros((new_detector.row, new_detector.col), dtype=float)
    # charge[pix_rows, pix_cols] = new_detector.pixels.get_pixel_charges()

    return new_detector

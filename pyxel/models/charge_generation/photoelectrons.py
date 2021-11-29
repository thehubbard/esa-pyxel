#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Simple model to convert photon into photo-electrons inside detector."""
import numpy as np

from pyxel.detectors import Detector

# TODO: docstring, private function, what is eta, characteristics
# TODO: put code from qe_map here and use qe as model parameter


# TODO: Fix this
# @validators.validate
# @config.argument(name='', label='', units='', validate=)
def simple_conversion(detector: Detector) -> None:
    """Generate charge from incident photon via photoelectric effect, simple statistical model.

    Parameters
    ----------
    detector : Detector
        Pyxel Detector object.
    """
    geo = detector.geometry
    ch = detector.characteristics
    ph = detector.photon

    detector_charge = np.zeros(
        (geo.row, geo.col)
    )  # all pixels has zero charge by default
    photon_rows, photon_cols = ph.array.shape
    detector_charge[slice(0, photon_rows), slice(0, photon_cols)] = (
        ph.array * ch.qe * ch.eta
    )
    detector.charge.add_charge_array(detector_charge.astype(int))


# # TODO: Fix this
# # @validators.validate
# # @config.argument(name='', label='', units='', validate=)
# def monte_carlo_conversion(detector: Detector) -> None:
#     """Generate charge from incident photon via photoelectric effect, more exact, stochastic (Monte Carlo) model.
#
#     :param detector: Pyxel Detector object
#     """
#     logging.info("")
#
#     # detector.qe <= 1
#     # detector.eta <= 1
#     # if np.random.rand(size) <= detector.qe:
#     #     pass    # 1 e
#     # else:
#     #     pass
#     # if np.random.rand(size) <= detector.eta:
#     #     pass    # 1 e
#     # else:
#     #     pass
#     # TODO: random number for QE
#     # TODO: random number for eta
#     # TODO: energy threshold
#
#
# def random_pos(detector: Detector) -> None:
#     """Generate random position for photoelectric effect inside detector.
#
#     :param detector: Pyxel Detector object
#     """
#     # pos1 = detector.vert_dimension * np.random.random()
#     # pos2 = detector.horz_dimension * np.random.random()
#
#     # size = 0
#     # pos3 = -1 * detector.total_thickness * np.random.rand(size)
#     # return pos3
#     raise NotImplementedError

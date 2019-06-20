#   --------------------------------------------------------------------------
#   Copyright 2018 SCI-FIV, ESA (European Space Agency)
#   --------------------------------------------------------------------------
"""Simple model to load charge profiles."""
import logging
import numpy as np
import pyxel
from pyxel.detectors.detector import Detector


@pyxel.validate
@pyxel.argument(name='txt_file', label='file path', units='', validate=pyxel.check_path)
def charge_profile(detector: Detector,
                   txt_file: str):
    """Load charge profile from txt file for detector, mostly for but not limited to CCDs.

    :param detector: Pyxel Detector object
    :param txt_file: file path
    """
    logger = logging.getLogger('pyxel')
    logger.info('')
    geo = detector.geometry

    init_ver_position = np.arange(0.0, geo.row, 1.0) * geo.pixel_vert_size
    init_hor_position = np.arange(0.0, geo.col, 1.0) * geo.pixel_horz_size
    init_ver_position = np.repeat(init_ver_position, geo.col)
    init_hor_position = np.tile(init_hor_position, geo.row)

    charge_from_file = np.loadtxt(txt_file, ndmin=2)
    charge_number = charge_from_file.flatten()
    where_non_zero = np.where(charge_number > 0.)
    charge_number = charge_number[where_non_zero]
    init_ver_position = init_ver_position[where_non_zero]
    init_hor_position = init_hor_position[where_non_zero]
    size = charge_number.size
    init_ver_position += np.random.rand(size) * geo.pixel_vert_size
    init_hor_position += np.random.rand(size) * geo.pixel_horz_size

    detector.charge.add_charge(particle_type='e',
                               particles_per_cluster=charge_number,
                               init_energy=[0.] * size,
                               init_ver_position=init_ver_position,
                               init_hor_position=init_hor_position,
                               init_z_position=[0.] * size,
                               init_ver_velocity=[0.] * size,
                               init_hor_velocity=[0.] * size,
                               init_z_velocity=[0.] * size)

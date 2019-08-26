"""Pyxel util functions for Particle classes."""
import math
import numpy as np
import typing as t

if t.TYPE_CHECKING:
    from ..detectors import Detector


def check_energy(initial_energy: t.Union[int, float]) -> None:
    """Check energy of the particle if it is a float or int.

    :param initial_energy:
    :return:
    """
    # FRED: Flatten this
    if isinstance(initial_energy, int) or isinstance(initial_energy, float):
        pass
    else:
        raise ValueError('Given particle energy could not be read')


def check_position(detector: "Detector", initial_position: t.Tuple[float, float, float]) -> None:
    """Check position of the particle if it is a numpy array and inside the detector.

    :param detector:
    :param initial_position:
    :return:
    """
    if not isinstance(initial_position, np.ndarray):
        raise ValueError('Position of particle is not a numpy array (int or float)')

    if not (0.0 <= initial_position[0] <= detector.geometry.vert_dimension):
        raise ValueError('Vertical position of particle is outside the detector')

    if not (0.0 <= initial_position[1] <= detector.geometry.horz_dimension):
        raise ValueError('Horizontal position of particle is outside the detector')

    if not (-1 * detector.geometry.total_thickness <= initial_position[2] <= 0.0):
        raise ValueError('Z position of particle is outside the detector')


def random_direction(v_abs: float = 1.0) -> np.ndarray:    # TODO check random angles and direction
    """Generate random direction for a photon.

    :param v_abs:
    :return:
    """
    alpha = 2 * math.pi * np.random.random()
    beta = 2. * math.pi * np.random.random()
    v_z = v_abs * math.sin(alpha)
    v_ver = v_abs * math.cos(alpha) * math.cos(beta)
    v_hor = v_abs * math.cos(alpha) * math.sin(beta)
    return np.array([v_ver, v_hor, v_z])

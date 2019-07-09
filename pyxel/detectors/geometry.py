"""Geometry class for detector."""
import esapy_config.config as ec
from esapy_config import validators


@ec.config()
class Geometry:
    """Geometrical attributes of the detector."""

    row = ec.setting(
        type=int,
        default=0,
        validator=validators.interval(0, 10000),
        doc='Number of pixel rows'
    )
    col = ec.setting(
        type=int,
        default=0,
        validator=validators.interval(0, 10000),
        doc='Number of pixel columns'
    )
    total_thickness = ec.setting(
        type=float,
        default=0.0,
        validator=validators.interval(0, 10000),
        metadata={'units': 'um'},
        doc='Thickness of detector'
    )
    pixel_vert_size = ec.setting(
        type=float,
        default=0.0,
        validator=validators.interval(0, 1000),
        metadata={'units': 'um'},
        doc='Vertical dimension of pixel'
    )
    pixel_horz_size = ec.setting(
        type=float,
        default=0.0,
        validator=validators.interval(0, 1000),
        metadata={'units': 'um'},
        doc='Horizontal dimension of pixel'
    )

    @property
    def horz_dimension(self) -> float:
        """Get total horizontal dimension of detector. Calculated automatically.

        :return: horizontal dimension
        """
        return self.pixel_horz_size * self.col

    @property
    def vert_dimension(self) -> float:
        """Get total vertical dimension of detector. Calculated automatically.

        :return: vertical dimension
        """
        return self.pixel_vert_size * self.row

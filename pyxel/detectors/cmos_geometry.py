"""TBW."""
from pyxel.detectors.geometry import Geometry
import esapy_config.config as ec
from esapy_config import validators


@ec.config(mode='RO')
class CMOSGeometry(Geometry):
    """Geometrical attributes of a CMOS-based detector."""

    n_output = ec.setting(
        type=int,
        default=1,
        validator=validators.interval(0, 32),
        doc='Number of detector outputs'
    )
    n_row_overhead = ec.setting(
        type=int,
        default=0,
        validator=validators.interval(0, 100),
        doc='New row overhead in pixel'
        # This allows for a short wait at the end of a row before starting the next row.
    )
    n_frame_overhead = ec.setting(
        type=int,
        default=0,
        validator=validators.interval(0, 100),
        doc='New frame overhead in rows'
        # This allows for a short wait at the end of a frame before starting the next frame.
    )
    reverse_scan_direction = ec.setting(
        type=bool,
        default=False,
        doc='Set this True to reverse the fast scanner readout directions. '
            'This capability was added to support Teledyne’s programmable fast scan readout directions. '
            'The default setting (False) corresponds to what HxRG detectors default to upon power up.'
    )
    reference_pixel_border_width = ec.setting(
        type=int,
        default=4,
        validator=validators.interval(0, 32),
        doc='Width of reference pixel border around image area'
    )

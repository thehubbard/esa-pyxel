from . import Geometry

class CMOSGeometry(Geometry):
    def __init__(
        self,
        n_output: int = 1,
        n_row_overhead: int = 0,
        n_frame_overhead: int = 0,
        reverse_scan_direction: bool = False,
        reference_pixel_border_width: int = 4,
    ): ...
    @property
    def n_output(self) -> int: ...
    @property
    def n_row_overhead(self) -> int: ...
    @property
    def n_frame_overhead(self) -> int: ...
    @property
    def reverse_scan_direction(self) -> bool: ...
    @property
    def reference_pixel_border_width(self) -> int: ...

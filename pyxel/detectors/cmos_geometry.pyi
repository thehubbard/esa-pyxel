from . import Geometry

class CMOSGeometry(Geometry):
    def __init__(
        self,
        n_output: int = ...,
        n_row_overhead: int = ...,
        n_frame_overhead: int = ...,
        reverse_scan_direction: bool = ...,
        reference_pixel_border_width: int = ...,
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

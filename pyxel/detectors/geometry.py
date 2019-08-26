"""Geometry class for detector."""
import numpy as np


class Geometry:
    """Geometrical attributes of the detector."""

    def __init__(
        self,
        row: int = 0,
        col: int = 0,
        total_thickness: float = 0.0,  # unit: um
        pixel_vert_size: float = 0.0,  # unit: um
        pixel_horz_size: float = 0.0,  # unit: um
    ):
        """Create a new instance of `Geometry`.

        Parameters
        ----------
        row: int
            Number of pixel rows.
        col: int
            Number of pixel columns.
        total_thickness: float
            Thickness of detector. Unit: um
        pixel_vert_size: float
            Vertical dimension of pixel. Unit: um
        pixel_horz_size: float
            Horizontal dimension of pixel. Unit: um
        """
        if row not in range(10001):
            raise ValueError("'row' must be between 0 and 10000.")

        if col not in range(10001):
            raise ValueError("'col' must be between 0 and 10000.")

        if not (0.0 <= total_thickness <= 10000.0):
            raise ValueError("'total_thickness' must be between 0.0 and 10000.0.")

        if not (0.0 <= pixel_vert_size <= 1000.0):
            raise ValueError("'pixel_vert_size' must be between 0.0 and 1000.0.")

        if not (0.0 <= pixel_horz_size <= 1000.0):
            raise ValueError("'pixel_horz_size' must be between 0.0 and 1000.0.")

        self._row = row
        self._col = col
        self._total_thickness = total_thickness
        self._pixel_vert_size = pixel_vert_size
        self._pixel_horz_size = pixel_horz_size

    @property
    def row(self) -> int:
        """Get Number of pixel rows."""
        return self._row

    @row.setter
    def row(self, value: int):
        """Set Number of pixel rows."""
        if value not in range(10001):
            raise ValueError("'row' must be between 0 and 10000.")

        self._row = value

    @property
    def col(self) -> int:
        """Get Number of pixel columns."""
        return self._col

    @col.setter
    def col(self, value: int):
        """Set Number of pixel columns."""
        if value not in range(10001):
            raise ValueError("'columns' must be between 0 and 10000.")

        self._col = value

    @property
    def total_thickness(self) -> float:
        """Get Thickness of detector."""
        return self._total_thickness

    @total_thickness.setter
    def total_thickness(self, value: float):
        """Set Thickness of detector."""
        if not (0.0 <= value <= 10000.0):
            raise ValueError("'total_thickness' must be between 0.0 and 10000.0.")

        self._total_thickness = value

    @property
    def pixel_vert_size(self) -> float:
        """Get Vertical dimension of pixel."""
        return self._pixel_vert_size

    @pixel_vert_size.setter
    def pixel_vert_size(self, value: float):
        """Set Vertical dimension of pixel."""
        if not (0.0 <= value <= 1000.0):
            raise ValueError("'pixel_vert_size' must be between 0.0 and 1000.0.")

        self._pixel_vert_size = value

    @property
    def pixel_horz_size(self) -> float:
        """Get Horizontal dimension of pixel."""
        return self._pixel_horz_size

    @pixel_horz_size.setter
    def pixel_horz_size(self, value: float):
        """Set Horizontal dimension of pixel."""
        if not (0.0 <= value <= 1000.0):
            raise ValueError("'pixel_horz_size' must be between 0.0 and 1000.0.")

        self._pixel_horz_size = value

    @property
    def horz_dimension(self) -> float:
        """Get total horizontal dimension of detector. Calculated automatically.

        Return
        ------
        float
            horizontal dimension
        """
        return self.pixel_horz_size * self.col

    @property
    def vert_dimension(self) -> float:
        """Get total vertical dimension of detector. Calculated automatically.

        Return
        ------
        float
            vertical dimension
        """
        return self.pixel_vert_size * self.row

    def vertical_pixel_center_pos_list(self) -> np.ndarray:
        """Generate horizontal position list of all pixel centers in detector imaging area."""
        init_ver_position = np.arange(0.0, self.row, 1.0) * self.pixel_vert_size
        init_ver_position += self.pixel_vert_size / 2.0
        return np.repeat(init_ver_position, self.col)

    def horizontal_pixel_center_pos_list(self) -> np.ndarray:
        """Generate horizontal position list of all pixel centers in detector imaging area."""
        init_hor_position = np.arange(0.0, self.col, 1.0) * self.pixel_horz_size
        init_hor_position += self.pixel_horz_size / 2.0
        return np.tile(init_hor_position, self.row)

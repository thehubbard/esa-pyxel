"""
Geometry class for detector
"""


class CMOSGeometry:

    def __init__(self,
                 row: int = None, col: int = None,
                 depletion_thickness: float = None,
                 field_free_thickness: float = None,
                 total_thickness: float = None,
                 pixel_vert_size: float = None,
                 pixel_horz_size: float = None,
                 material: str = None,
                 n_acceptor: float = None,
                 n_donor: float = None,
                 bias_voltage: float = None,
                 n_output: int = None,
                 n_row_overhead: int = None,
                 n_frame_overhead: int = None,
                 reverse_scan_direction: bool = None,
                 reference_pixel_border_width: int = None) -> None:
        """
        Initialize the CMOS geometry
        :param row:
        :param col:
        :param depletion_thickness:
        :param field_free_thickness:
        :param total_thickness:
        :param pixel_vert_size:
        :param pixel_horz_size:
        :param material:
        :param n_acceptor:
        :param n_donor:
        :param bias_voltage:
        :param n_output:
        :param n_row_overhead:
        :param n_frame_overhead:
        :param reverse_scan_direction:
        :param reference_pixel_border_width:
        """
        self.row = row
        self.col = col
        self.total_thickness = total_thickness                  # TODO: add units
        self.depletion_thickness = depletion_thickness          # TODO: calculate this or get from config if defined
        self.field_free_thickness = field_free_thickness        # TODO: calculate this or get from config if defined
        self.pixel_vert_size = pixel_vert_size
        self.pixel_horz_size = pixel_horz_size

        self.n_acceptor = n_acceptor
        self.n_donor = n_donor
        self.bias_voltage = bias_voltage

        self.material_density = None
        self.material_ionization_energy = None
        self.band_gap = None
        self.horz_dimension = None
        self.vert_dimension = None

        self.calculate_geometry_parameters()
        self.set_material(material)

        # CMOS specific geometry parameters
        self.n_row_overhead = n_row_overhead
        self.n_frame_overhead = n_frame_overhead
        self.n_output = n_output
        self.reverse_scan_direction = reverse_scan_direction
        self.reference_pixel_border_width = reference_pixel_border_width

    def set_material(self, material):
        """
        Set material properties
        :param material:
        :return:
        """

        if material == 'silicon' or 'Si' or 'si':
            self.material_density = 2.328               # TODO add unit (g/cm3)
            self.material_ionization_energy = 3.6       # TODO add unit (eV)
            self.band_gap = 1.12                        # TODO add unit (eV)

        else:
            raise NotImplementedError('Given material has not implemented yet')

    def calculate_geometry_parameters(self):
        """
        Calculate and update missing geometry parameters from other provided by the user
        :return:
        """

        self.horz_dimension = self.pixel_horz_size * self.col
        self.vert_dimension = self.pixel_vert_size * self.row

    def calculate_field_free_thickness(self):
        pass

    def calculate_depletion_thickness(self):
        pass

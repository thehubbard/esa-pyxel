import typing as t

M_ELECTRON = 9.10938356e-31  # kg     # TODO put these global constants to a data file

class Material:
    def __init__(
        self,
        trapped_charge: t.Optional[str] = None,
        n_acceptor: float = 0.0,
        n_donor: float = 0.0,
        material: str = "silicon",
        material_density: float = 2.328,
        ionization_energy: float = 3.6,
        band_gap: float = 1.12,
        e_effective_mass: float = 0.5 * M_ELECTRON,
    ): ...
    @property
    def trapped_charge(self) -> t.Optional[str]: ...
    @property
    def n_acceptor(self) -> float: ...
    @property
    def n_donor(self) -> float: ...
    @property
    def material(self) -> str: ...
    @property
    def material_density(self) -> float: ...
    @property
    def ionization_energy(self) -> float: ...
    @property
    def band_gap(self) -> float: ...
    @property
    def e_effective_mass(self) -> float: ...

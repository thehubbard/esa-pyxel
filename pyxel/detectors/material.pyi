import typing as t

class Material:
    def __init__(
        self,
        trapped_charge: t.Optional[str] = ...,
        n_acceptor: float = ...,
        n_donor: float = ...,
        material: str = "silicon",
        material_density: float = ...,
        ionization_energy: float = ...,
        band_gap: float = ...,
        e_effective_mass: float = ...,
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

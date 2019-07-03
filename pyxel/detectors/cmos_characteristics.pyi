from . import Characteristics

class CMOSCharacteristics(Characteristics):
    def __init__(
        self,
        cutoff: float = 2.5,
        vbiaspower: float = 3.350,
        dsub: float = 0.5,
        vreset: float = 0.250,
        biasgate: float = 2.300,
        preampref: float = 1.700,
    ): ...
    @property
    def cutoff(self) -> float: ...
    @property
    def vbiaspower(self) -> float: ...
    @property
    def dsub(self) -> float: ...
    @property
    def vreset(self) -> float: ...
    @property
    def biasgate(self) -> float: ...
    @property
    def preampref(self) -> float: ...

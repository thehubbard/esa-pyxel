from . import Characteristics

class CMOSCharacteristics(Characteristics):
    def __init__(
        self,
        cutoff: float = ...,
        vbiaspower: float = ...,
        dsub: float = ...,
        vreset: float = ...,
        biasgate: float = ...,
        preampref: float = ...,
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

"""TBW."""
from pyxel.detectors.characteristics import Characteristics


# TODO: 'svg' should be the full volume and not the half
class CCDCharacteristics(Characteristics):
    """Characteristical attributes of a CCD detector."""

    def __init__(
        self,

        # Parameters for `Characteristics`
        qe: float = 0.0,  # unit: NA
        eta: float = 0.0,  # unit: electron/photon
        sv: float = 0.0,  # unit: volt/electron
        amp: float = 0.0,  # unit: V/V
        a1: float = 0.0,  # unit: V/V
        a2: int = 0,  # unit: adu/V
        fwc: int = 0,  # unit: electron
        vg: float = 0.0,  # unit: cm^2
        dt: float = 0.0,  # unit: s

        # Parameters specific `CCDCharacteristics`
        fwc_serial: int = 0,  # unit: electron
        svg: float = 0.0,  # unit: cm^2
        t: float = 0.0,  # unit: s
        st: float = 0.0,  # unit: s
    ):
        """Create an instance of `CCDCharacteristics`.

        Parameters
        ----------
        fwc_serial: int
            Full well capacity (serial register). Unit: electron
        svg: float
            Half pixel volume charge can occupy (serial register). Unit: cm^2
        t: float
            Parallel transfer period. Unit: s
        st: float
            Serial transfer period. Unit: s
        """
        if fwc_serial not in range(10000001):
            raise ValueError("'fwc_serial' must be between 0 and 1e+7.")

        if not (0.0 <= svg <= 1.0):
            raise ValueError("'svg' must be between 0.0 and 1.0.")

        if not (0.0 <= t <= 10.0):
            raise ValueError("'t' must be between 0.0 and 10.0.")

        if not (0.0 <= st <= 10.0):
            raise ValueError("'st' must be between 0.0 and 10.0.")

        super().__init__(
            qe=qe, eta=eta, sv=sv, amp=amp, a1=a1, a2=a2, fwc=fwc, vg=vg, dt=dt
        )
        self._fwc_serial = fwc_serial
        self._svg = svg
        self._t = t
        self._st = st

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"CCDCharacteristics(fwc_serial={self.fwc_serial}, svg={self.svg}, t={self.t}, st={self.st})"

    @property
    def fwc_serial(self) -> int:
        """Get Full well capacity (serial register)."""
        return self._fwc_serial

    @fwc_serial.setter
    def fwc_serial(self, value: int) -> None:
        """Set Full well capacity (serial register)."""
        if value not in range(10000001):
            raise ValueError("'fwc_serial' must be between 0 and 1e+7.")

        self._fwc_serial = value

    @property
    def svg(self) -> float:
        """Get Half pixel volume charge can occupy (serial register)."""
        return self._svg

    @svg.setter
    def svg(self, value: float) -> None:
        """Set Half pixel volume charge can occupy (serial register)."""
        if not (0.0 <= value <= 1.0):
            raise ValueError("'svg' must be between 0.0 and 1.0.")

        self._svg = value

    @property
    def t(self) -> float:
        """Get Parallel transfer period."""
        return self._t

    @t.setter
    def t(self, value: float) -> None:
        """Set Parallel transfer period."""
        if not (0.0 <= value <= 10.0):
            raise ValueError("'t' must be between 0.0 and 10.0.")

        self._t = value

    @property
    def st(self) -> float:
        """Set Serial transfer period."""
        return self._st

    @st.setter
    def st(self, value: float) -> None:
        """Get Serial transfer period."""
        if not (0.0 <= value <= 10.0):
            raise ValueError("'st' must be between 0.0 and 10.0.")

        self._st = value

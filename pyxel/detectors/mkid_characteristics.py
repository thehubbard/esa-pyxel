#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""
from pyxel.detectors import Characteristics


class MKIDCharacteristics(Characteristics):
    """Characteristical attributes of a MKID-based detector.

    Parameters
    ----------
    cutoff : float
        Cutoff wavelength. Unit: um
    vbiaspower : float
        VBIASPOWER. Unit: V
    dsub : float
        DSUB. Unit: V
    vreset : float
        VRESET. Unit: V
    biasgate : float
        BIASGATE. Unit: V
    preampref : float
        PREAMPREF. Unit: V
    tau_0 : float
        Material dependent characteristic time for the electron-phonon coupling. Unit: s
    n_0 : float
        Material dependent single spin density of states at
        the Fermi-level. Unit: um^-3 eV^-1
    t_c : float
        Material dependent critical temperature. Unit: K
    v : float
        Superconducting volume. Unit: um^3
    t_op : float
        Temperature. Unit: K
    tau_pb : float
        Phonon pair-breaking time. Unit: s
    tau_esc : float
        Phonon escape time. Unit: s
    tau_sat : float
        Saturation time. Unit: s
    Notes
    -----
    For the characteristics of aluminium, see :cite:p:`PhysRevB.104.L180506`.
    """

    def __init__(
        self,
        # Parameters for `Characteristics`
        qe: float = 0.0,
        eta: float = 0.0,
        sv: float = 0.0,
        amp: float = 0.0,
        a1: float = 0.0,
        a2: int = 0,
        fwc: int = 0,
        dt: float = 0.0,
        # Parameters specific `CMOSCharacteristics`
        cutoff: float = 2.5,  # unit: um
        vbiaspower: float = 3.35,  # unit: V
        dsub: float = 0.5,  # unit: V
        vreset: float = 0.25,  # unit: V
        biasgate: float = 2.3,  # unit: V
        preampref: float = 1.7,  # unit: V
        # Parameters specific `MKIDCharacteristics`
        tau_0: float = 4.4 * 1.0e-7,  # [s] (material-dependent)
        n_0: float = 1.72 * 1.0e10,  # [um^-3 eV^-1] (material-dependent)
        t_c: float = 1.26,  # [K] (material-dependent)
        v: float = 30.0,  # [um^3]
        t_op: float = 0.3,  # [K]
        tau_pb: float = 2.8 * 1.0e-10,  # [s]
        tau_esc: float = 1.4 * 1.0e-10,  # [s]
        tau_sat: float = 1.0e-3,  # [s]
    ):
        if not (1.7 <= cutoff <= 15.0):
            raise ValueError("'cutoff' must be between 1.7 and 15.0.")

        if not (0.0 <= vbiaspower <= 3.4):
            raise ValueError("'vbiaspower' must be between 0.0 and 3.4.")

        if not (0.3 <= dsub <= 1.0):
            raise ValueError("'dsub' must be between 0.3 and 1.0.")

        if not (0.0 <= vreset <= 0.3):
            raise ValueError("'vreset' must be between 0.0 and 0.3.")

        if not (1.8 <= biasgate <= 2.6):
            raise ValueError("'biasgate' must be between 1.8 and 2.6.")

        if not (0.0 <= preampref <= 4.0):
            raise ValueError("'preampref' must be between 0.0 and 4.0.")

        super().__init__(qe=qe, eta=eta, sv=sv, amp=amp, a1=a1, a2=a2, fwc=fwc, dt=dt)
        self._cutoff = cutoff
        self._vbiaspower = vbiaspower
        self._dsub = dsub
        self._vreset = vreset
        self._biasgate = biasgate
        self._preampref = preampref

        # TODO: validated the following values
        self._tau_0 = tau_0
        self._n_0 = n_0
        self._t_c = t_c
        self._v = v
        self._t_op = t_op
        self._tau_pb = tau_pb
        self._tau_esc = tau_esc
        self._tau_sat = tau_sat

    @property
    def cutoff(self) -> float:
        """Get Cutoff wavelength."""
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value: float) -> None:
        """Set Cutoff wavelength."""
        if not (1.7 <= value <= 15.0):
            raise ValueError("'cutoff' must be between 1.7 and 15.0.")

        self._cutoff = value

    @property
    def vbiaspower(self) -> float:
        """Get VBIASPOWER."""
        return self._vbiaspower

    @vbiaspower.setter
    def vbiaspower(self, value: float) -> None:
        """Set VBIASPOWER."""
        if not (0.0 <= value <= 3.4):
            raise ValueError("'vbiaspower' must be between 0.0 and 3.4.")

        self._vbiaspower = value

    @property
    def dsub(self) -> float:
        """Get DSUB."""
        return self._dsub

    @dsub.setter
    def dsub(self, value: float) -> None:
        """Set DSUB."""
        if not (0.3 <= value <= 1.0):
            raise ValueError("'dsub' must be between 0.3 and 1.0.")

        self._dsub = value

    @property
    def vreset(self) -> float:
        """Get VREST."""
        return self._vreset

    @vreset.setter
    def vreset(self, value: float) -> None:
        """Set VRESET."""
        if not (0.0 <= value <= 0.3):
            raise ValueError("'vreset' must be between 0.0 and 0.3.")

        self._vreset = value

    @property
    def biasgate(self) -> float:
        """Get BIASGATE."""
        return self._biasgate

    @biasgate.setter
    def biasgate(self, value: float) -> None:
        """Set BIASGATE."""
        if not (1.8 <= value <= 2.6):
            raise ValueError("'biasgate' must be between 1.8 and 2.6.")

        self._biasgate = value

    @property
    def preampref(self) -> float:
        """Get PREAMPREF."""
        return self._preampref

    @preampref.setter
    def preampref(self, value: float) -> None:
        """Set PREAMPREF."""
        if not (0.0 <= value <= 4.0):
            raise ValueError("'preampref' must be between 0.0 and 4.0.")

        self._preampref = value

    @property
    def tau_0(self) -> float:
        """Get characteristic time for the electron-phonon coupling."""
        return self._tau_0

    @tau_0.setter
    def tau_0(self, value) -> None:
        """Set characteristic time for the electron-phonon coupling."""
        # TODO: check if 'value' is valid
        self._tau_0 = value

    @property
    def n_0(self) -> float:
        """Get single spin density of states at the Fermi-level."""
        return self._n_0

    @n_0.setter
    def n_0(self, value) -> None:
        """Set single spin density of states at the Fermi-level."""
        # TODO: check if 'value' is valid
        self._n_0 = value

    @property
    def t_c(self) -> float:
        """Get critical temperature."""
        return self._t_c

    @t_c.setter
    def t_c(self, value) -> None:
        """Set critical temperature."""
        # TODO: check if 'value' is valid
        self._t_c = value

    @property
    def v(self) -> float:
        """Get superconducting volume."""
        return self._v

    @v.setter
    def v(self, value) -> None:
        """Set superconducting volume."""
        # TODO: check if 'value' is valid
        self._v = value

    @property
    def t_op(self) -> float:
        """Get temperature."""
        return self._t_op

    @t_op.setter
    def t_op(self, value) -> None:
        """Set temperature."""
        # TODO: check if 'value' is valid
        self._t_op = value

    @property
    def tau_pb(self) -> float:
        """Get phonon pair-breaking time."""
        return self._tau_pb

    @tau_pb.setter
    def tau_pb(self, value) -> None:
        """Set phonon pair-breaking time."""
        # TODO: check if 'value' is valid
        self._tau_pb = value

    @property
    def tau_esc(self) -> float:
        """Get phonon escape time."""
        return self._tau_esc

    @tau_esc.setter
    def tau_esc(self, value) -> None:
        """Set phonon escape time."""
        # TODO: check if 'value' is valid
        self._tau_esc = value

    @property
    def tau_sat(self) -> float:
        """Get saturation time."""
        return self._tau_sat

    @tau_sat.setter
    def tau_sat(self, value) -> None:
        """Set saturation time."""
        # TODO: check if 'value' is valid
        self._tau_sat = value

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""
import typing as t

import numpy as np

from pyxel.util.memory import get_size


# TODO: 'vg' should be the full volume and not the half
class Characteristics:
    """Characteristical attributes of the detector."""

    def __init__(
        self,
        qe: float = 1.0,  # unit: NA
        eta: float = 1.0,  # unit: electron/photon
        sv: float = 1.0,  # unit: volt/electron
        amp: float = 1.0,  # unit: V/V
        a1: float = 1.0,  # unit: V/V
        a2: float = 1,  # unit: adu/V
        fwc: float = 0,  # unit: electron
        dt: float = 0.0,  # unit: s
    ):
        """Create an instance of `Characteristics`.

        Parameters
        ----------
        qe: float
            Quantum efficiency.
        eta: float
            Quantum yield. Unit: e-/photon
        sv: float
            Sensitivity of charge readout. Unit: V/e-
        amp: float
            Gain of output amplifier. Unit: V/V
        a1: float
            Gain of the signal processor. Unit: V/V
        a2: int
            Gain of the Analog-Digital Converter. Unit: ADU/V
        fwc: int
            Full well capacity. Unit: e-
        dt: float
            Pixel dwell time. Unit: s
        """
        if not (0.0 <= qe <= 1.0):
            raise ValueError("'qe' must be between 0.0 and 1.0.")

        if not (0.0 <= eta <= 1.0):
            raise ValueError("'eta' must be between 0.0 and 1.0.")

        if not (0.0 <= sv <= 100.0):
            raise ValueError("'sv' must be between 0.0 and 100.0.")

        if not (0.0 <= amp <= 100.0):
            raise ValueError("'amp' must be between 0.0 and 100.0.")

        if not (0.0 <= a1 <= 100.0):
            raise ValueError("'a1' must be between 0.0 and 100.0.")

        if a2 not in range(65537):
            raise ValueError("'a2' must be between 0 and 65536.")

        if not (0.0 <= fwc <= 1.0e7):
            raise ValueError("'fwc' must be between 0 and 1e7.")

        if not (0.0 <= dt <= 10.0):
            raise ValueError("'dt' must be between 0.0 and 10.0.")

        self._qe = qe  # type: t.Union[float, np.ndarray]
        self._eta = eta
        self._sv = sv
        self._amp = amp
        self._a1 = a1
        self._a2 = a2
        self._fwc = fwc
        self._dt = dt

        self._numbytes = 0

    @property
    def qe(self) -> t.Union[float, np.ndarray]:
        """Get Quantum efficiency."""
        return self._qe

    @qe.setter
    def qe(self, value: t.Union[float, np.ndarray]) -> None:
        """Set Quantum efficiency."""
        if np.min(value) < 0.0 and np.max(value) <= 1.0:
            raise ValueError("'QE' values must be between 0.0 and 1.0.")

        self._qe = value

    @property
    def eta(self) -> float:
        """Get Quantum yield."""
        return self._eta

    @eta.setter
    def eta(self, value: float) -> None:
        """Set Quantum yield."""
        if not (0.0 <= value <= 1.0):
            raise ValueError("'eta' must be between 0.0 and 1.0.")

        self._eta = value

    @property
    def sv(self) -> float:
        """Get Sensitivity of charge readout."""
        return self._sv

    @sv.setter
    def sv(self, value: float) -> None:
        """Set Sensitivity of charge readout."""
        if not (0.0 <= value <= 100.0):
            raise ValueError("'sv' must be between 0.0 and 100.0.")
        self._sv = value

    @property
    def amp(self) -> float:
        """Get Gain of output amplifier."""
        return self._amp

    @amp.setter
    def amp(self, value: float) -> None:
        """Set Gain of output amplifier."""
        if not (0.0 <= value <= 100.0):
            raise ValueError("'amp' must be between 0.0 and 100.0.")

        self._amp = value

    @property
    def a1(self) -> float:
        """Get Gain of the signal processor."""
        return self._a1

    @a1.setter
    def a1(self, value: float) -> None:
        """Set Gain of the signal processor."""
        if not (0.0 <= value <= 100.0):
            raise ValueError("'a1' must be between 0.0 and 100.0.")

        self._a1 = value

    @property
    def a2(self) -> float:
        """Get Gain of the Analog-Digital Converter."""
        return self._a2

    @a2.setter
    def a2(self, value: float) -> None:
        """Set Gain of the Analog-Digital Converter."""
        if value not in range(65537):
            raise ValueError("'a2' must be between 0 and 65536.")

        self._a2 = value

    @property
    def fwc(self) -> float:
        """Get Full well capacity."""
        return self._fwc

    @fwc.setter
    def fwc(self, value: float) -> None:
        """Set Full well capacity."""
        if value not in range(10000001):
            raise ValueError("'fwc' must be between 0 and 1e+7.")

        self._fwc = value

    @property
    def dt(self) -> float:
        """Get Pixel dwell time."""
        return self._dt

    @dt.setter
    def dt(self, value: float) -> None:
        """Set Pixel dwell time."""
        if not (0.0 <= value <= 10.0):
            raise ValueError("'dt' must be between 0.0 and 10.0.")

        self._dt = value

    @property
    def numbytes(self) -> int:
        """Recursively calculates object size in bytes using Pympler library.

        Returns
        -------
        int
            Size of the object in bytes.
        """
        self._numbytes = get_size(self)
        return self._numbytes

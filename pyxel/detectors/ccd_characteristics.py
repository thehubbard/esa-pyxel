#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""
from pyxel.detectors import Characteristics


# TODO: 'svg' should be the full volume and not the half
class CCDCharacteristics(Characteristics):
    """Characteristical attributes of a CCD detector."""

    def __init__(
        self,
        # Parameters for `Characteristics`
        qe: float = 1.0,  # unit: NA
        eta: float = 1.0,  # unit: electron/photon
        sv: float = 1.0,  # unit: volt/electron
        amp: float = 1.0,  # unit: V/V
        a1: float = 1.0,  # unit: V/V
        a2: int = 1,  # unit: adu/V
        fwc: int = 0,  # unit: electron
        dt: float = 0.0,  # unit: s
    ):
        """Create an instance of `CCDCharacteristics`."""
        super().__init__(qe=qe, eta=eta, sv=sv, amp=amp, a1=a1, a2=a2, fwc=fwc, dt=dt)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__  # type: str
        return f"{cls_name}"

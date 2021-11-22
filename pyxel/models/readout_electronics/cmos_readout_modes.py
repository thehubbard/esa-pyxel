#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Non-Destructive Readout modes for CMOS-based detectors."""
import typing as t

from typing_extensions import Literal

from pyxel.detectors import CMOS


# TODO: find a way to integrate this in readout definition in running mode and remove this model for 1.0
# TODO: different readout noise for different readout modes?
def non_destructive_readout(
    detector: CMOS,
    mode: Literal["uncorrelated", "CDS", "Fowler-N", "UTR"],
    fowler_samples: t.Optional[int] = None,
) -> None:
    """Non-Destructive Readout modes for CMOS-based detectors.

    Parameters
    ----------
    detector : CMOS
        CMOS detector object.
    mode : str
        Valid values: 'uncorrelated', 'CDS', 'Fowler-N', 'UTR'
    fowler_samples : int

    Raises
    ------
    ValueError
        If the 'detector' is not properly initialized or
        if the parameters 'mode' and/or 'fowler_samples' are incorrect.
    TypeError
        If 'mode' is not recognized.
    """
    # Validation
    if mode == "Fowler-N":
        if fowler_samples is None:
            raise ValueError("Missing parameter 'fowler_samples' for mode 'Fowler-N'.")

        if fowler_samples < 1:
            raise ValueError("Parameter 'fowler_samples' must be > 1.")

    if mode != "Fowler-N" and fowler_samples is not None:
        raise ValueError(
            "Parameter 'fowler_samples' can only be used for mode 'Fowler-N'."
        )

    if not detector.non_destructive_readout or not detector.is_dynamic:
        raise ValueError(
            "Detector is must have a non-destructive readout and must be dynamic."
        )

    if not detector.times_linear:
        raise ValueError("Detector's time must be linear.")

    detector.read_out = False
    if mode == "uncorrelated":
        if detector.pipeline_count == detector.num_steps - 1:
            detector.read_out = True

    elif mode == "CDS":
        if detector.pipeline_count == 0 or detector.pipeline_count == (
            detector.num_steps - 1
        ):
            detector.read_out = True

    elif mode == "Fowler-N":
        if t.TYPE_CHECKING:
            # This 'assert' is only for Mypy.
            assert fowler_samples is not None

        nt = fowler_samples  # type: int
        detector.read_out = True
        if nt <= detector.pipeline_count < (detector.num_steps - nt):
            detector.read_out = False

    elif mode == "UTR":
        detector.read_out = True

    else:
        raise TypeError(f"Unknown mode {mode!r}.")

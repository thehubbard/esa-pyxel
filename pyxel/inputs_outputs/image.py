#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""

import typing as t
from pathlib import Path

import numpy as np
from astropy.io import fits


def load_image(filename: t.Union[str, Path]) -> np.ndarray:
    """
    TBW.

    Parameters
    ----------
    filename

    Returns
    -------
    array

    """

    filename_path = Path(filename).resolve()

    if not filename_path.exists():
        raise FileNotFoundError(f"Input file '{filename_path}' can not be found.")

    suffix = filename_path.suffix.lower()

    if suffix.startswith(".fits"):
        data_2d = fits.getdata(filename_path)  # type: np.ndarray

    elif suffix.startswith(".npy"):
        data_2d = np.load(filename_path)

    elif suffix.startswith(".txt") or suffix.startswith(".data"):
        # TODO: this is a convoluted implementation. Change to:
        # for sep in [' ', ',', '|', ';']:
        #     try:
        #         data = np.loadtxt(path, delimiter=sep[ii])
        #     except ValueError:
        #         pass
        #     else:
        #         break
        sep = [" ", ",", "|", ";"]
        ii, jj = 0, 1
        while jj:
            try:
                jj -= 1
                data_2d = np.loadtxt(filename_path, delimiter=sep[ii])
            except ValueError:
                ii += 1
                jj += 1
                if ii >= len(sep):
                    break

    else:
        raise NotImplementedError("Only .npy, .fits, .txt and .data implemented.")

    return data_2d

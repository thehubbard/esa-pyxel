#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Subpackage to load images."""

import typing as t
from pathlib import Path

import numpy as np
from astropy.io import fits


def load_image(filename: t.Union[str, Path]) -> np.ndarray:
    """Load a 2D image.

    Parameters
    ----------
    filename : str or Path
        Filename to read an image. '.fits', '.npy' and '.txt' are accepted.

    Returns
    -------
    array : ndarray
        A 2D array.

    Raises
    ------
    FileNotFoundError
        If an image is not found.
    NotImplementedError
        When the extension of the filename is unknown.

    Examples
    --------
    >>> from pyxel.inputs_outputs import load_image
    >>> load_image('frame.fits')
    array([[-0.66328494, -0.63205819, ...]])

    >>> load_image('another_frame.npy')
    array([[-1.10136521, -0.93890239, ...]])
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

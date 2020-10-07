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
from PIL import Image
import pandas as pd


def load_image(filename: t.Union[str, Path]) -> np.ndarray:
    """Load a 2D image.

    Parameters
    ----------
    filename : str or Path
        Filename to read an image.
        {.npy, .fits, .txt, .data, .jpg, .jpeg, .bmp, .png, .tiff} are accepted.

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

    >>> load_image('rgb_frame.jpg')
    array([[234, 211, ...]])
    """
    filename_path = Path(filename).expanduser().resolve()

    if not filename_path.exists():
        raise FileNotFoundError(f"Input file '{filename_path}' can not be found.")

    suffix = filename_path.suffix.lower()

    if suffix.startswith(".fits"):
        data_2d = fits.getdata(filename_path)  # type: np.ndarray

    elif suffix.startswith(".npy"):
        data_2d = np.load(filename_path)

    elif suffix.startswith(".txt") or suffix.startswith(".data"):
        for sep in ["\t", " ", ",", "|", ";"]:
            try:
                data_2d = np.loadtxt(filename_path, delimiter=sep)
            except ValueError:
                pass
            else:
                break

    elif suffix.startswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        image_2d = Image.open(filename_path)
        image_2d_converted = image_2d.convert("LA")  # RGB to grayscale conversion
        data_2d = np.array(image_2d_converted)[:, :, 0]

    else:
        raise NotImplementedError(
            """Image format not supported. List of supported image formats:
            .npy, .fits, .txt, .data, .jpg, .jpeg, .bmp, .png, .tiff."""
        )

    return data_2d


def load_table(filename: t.Union[str, Path]) -> np.ndarray:
    """Loads a table from a file and returns a numpy array.

    Parameters
    ----------
    filename: str or Path
        Filename to read the table.

    Returns
    -------
    table: ndarray

    Raises
    ------
    FileNotFoundError
        If an image is not found.
    NotImplementedError
        When the extension of the filename is unknown.

    """
    filename_path = Path(filename).expanduser().resolve()

    if not filename_path.exists():
        raise FileNotFoundError(f"Input file '{filename_path}' can not be found.")

    suffix = filename_path.suffix.lower()

    if suffix.startswith(".npy"):
        table = np.load(filename_path)

    elif suffix.startswith('.xlsx'):
        table = np.array(pd.read_excel(filename_path, header=None))

    elif suffix.startswith('.csv'):
        table = np.array(pd.read_csv(filename_path, header=None))

    elif suffix.startswith(".txt") or suffix.startswith(".data"):
        for sep in ["\t", " ", ",", "|", ";"]:
            try:
                table = np.array(pd.read_table(filename_path, header=None, delimiter=sep))
            except ValueError:
                pass
            else:
                break

    else:
        raise NotImplementedError("Only .npy, .xlsx, .csv, .txt and .data implemented.")

    return table
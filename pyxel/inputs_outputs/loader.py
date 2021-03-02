#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Subpackage to load images and tables."""

import typing as t
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
from astropy.io import fits
from PIL import Image


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
    ValueError
        When the extension of the filename is unknown or separator is not found.

    Examples
    --------
    >>> from pyxel.inputs_outputs import load_image
    >>> load_image("frame.fits")
    array([[-0.66328494, -0.63205819, ...]])

    >>> load_image("another_frame.npy")
    array([[-1.10136521, -0.93890239, ...]])

    >>> load_image("rgb_frame.jpg")
    array([[234, 211, ...]])
    """
    filename_path = Path(filename).expanduser().resolve()

    if not filename_path.exists():
        raise FileNotFoundError(f"Input file '{filename_path}' can not be found.")

    suffix = filename_path.suffix.lower()

    if suffix.startswith(".fits"):
        with fsspec.open(filename_path, mode="rb") as file_handler:
            data_2d = fits.getdata(file_handler)  # type: np.ndarray

    elif suffix.startswith(".npy"):
        with fsspec.open(filename_path, mode="rb") as file_handler:
            data_2d = np.load(file_handler)

    elif suffix.startswith(".txt") or suffix.startswith(".data"):
        for sep in ["\t", " ", ",", "|", ";"]:
            try:
                with fsspec.open(filename_path, mode="r") as file_handler:
                    data_2d = np.loadtxt(file_handler, delimiter=sep)
                break
            except ValueError:
                pass
        else:
            raise ValueError(
                f"Cannot find the separator for filename '{filename_path}'."
            )

    elif suffix.startswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        with fsspec.open(filename_path, mode="rb") as file_handler:
            image_2d = Image.open(file_handler)
            image_2d_converted = image_2d.convert("LA")  # RGB to grayscale conversion

        data_2d = np.array(image_2d_converted)[:, :, 0]

    else:
        raise ValueError(
            "Image format not supported. List of supported image formats: "
            ".npy, .fits, .txt, .data, .jpg, .jpeg, .bmp, .png, .tiff."
        )

    return data_2d


def load_table(filename: t.Union[str, Path]) -> pd.DataFrame:
    """Load a table from a file and returns a pandas dataframe. No header is expected in xlsx.

    Parameters
    ----------
    filename: str or Path
        Filename to read the table.
        {.npy, .xlsx, .csv, .txt., .data} are accepted.

    Returns
    -------
    table: DataFrame

    Raises
    ------
    FileNotFoundError
        If an image is not found.
    ValueError
        When the extension of the filename is unknown or separator is not found.

    """
    filename_path = Path(filename).expanduser().resolve()

    if not filename_path.exists():
        raise FileNotFoundError(f"Input file '{filename_path}' can not be found.")

    suffix = filename_path.suffix.lower()

    if suffix.startswith(".npy"):
        table = pd.DataFrame(np.load(filename_path), dtype="float")

    elif suffix.startswith(".xlsx"):
        table = pd.read_excel(filename_path, header=None, convert_float=False)

    elif suffix.startswith(".csv"):
        for sep in ["\t", " ", ",", "|", ";"]:
            try:
                # numpy will return ValueError with a wrong delimiter
                table = pd.read_csv(
                    filename_path, delimiter=sep, header=None, dtype="float"
                )
                break
            except ValueError:
                pass
        else:
            raise ValueError("Cannot find the separator.")

    elif suffix.startswith(".txt") or suffix.startswith(".data"):
        for sep in ["\t", " ", ",", "|", ";"]:
            try:
                table = pd.read_table(
                    filename_path, delimiter=sep, header=None, dtype="float"
                )
                break
            except ValueError:
                pass
        else:
            raise ValueError("Cannot find the separator.")

    else:
        raise ValueError("Only .npy, .xlsx, .csv, .txt and .data implemented.")

    return table

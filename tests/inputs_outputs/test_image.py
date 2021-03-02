#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import os
import re
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from PIL import Image
from pytest_httpserver import HTTPServer  # pip install pytest-httpserver

from pyxel.inputs_outputs import load_image, load_table


@pytest.fixture
def valid_multiple_hdus() -> fits.HDUList:
    """Create a valid HDUList with one 'PrimaryHDU' and one 'ImageHDU'."""
    # Create a new image
    primary_2d = np.array([[5, 6], [7, 8]], dtype=np.uint16)
    secondary_2d = np.array([[9, 10], [11, 12]], dtype=np.uint16)

    hdu_primary = fits.PrimaryHDU(primary_2d)
    hdu_secondary = fits.ImageHDU(secondary_2d)
    hdu_lst = fits.HDUList([hdu_primary, hdu_secondary])

    return hdu_lst


@pytest.fixture
def valid_table() -> pd.DataFrame:
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    return pd.DataFrame(array, dtype="float64")


@pytest.fixture
def valid_pil_image() -> Image.Image:
    """Create a valid RGB PIL image."""
    data_2d = np.array([[10, 20], [30, 40]], dtype="uint8")
    pil_image = Image.fromarray(data_2d).convert("RGB")

    return pil_image


@pytest.fixture
def valid_data2d_http_hostname(
    tmp_path: Path,
    httpserver: HTTPServer,
    valid_pil_image: Image.Image,
    valid_multiple_hdus: fits.HDUList,
) -> str:
    """Create valid 2D files on a temporary HTTP server."""
    # Get current folder
    current_folder = Path().cwd()  # type: Path

    try:
        os.chdir(tmp_path)

        # Create folder 'data'
        Path("data").mkdir(parents=True, exist_ok=True)

        data_2d = np.array([[1, 2], [3, 4]], dtype=np.uint16)

        # Save 2d images
        fits.writeto("data/img.fits", data=data_2d)
        valid_multiple_hdus.writeto("data/img_multiple.fits")
        np.save("data/img.npy", arr=data_2d)
        np.savetxt("data/img_tab.txt", X=data_2d, delimiter="\t")
        np.savetxt("data/img_space.txt", X=data_2d, delimiter=" ")
        np.savetxt("data/img_comma.txt", X=data_2d, delimiter=",")
        np.savetxt("data/img_pipe.txt", X=data_2d, delimiter="|")
        np.savetxt("data/img_semicolon.txt", X=data_2d, delimiter=";")

        valid_pil_image.save("data/img.jpg")
        valid_pil_image.save("data/img.jpeg")
        valid_pil_image.save("data/img.png")
        valid_pil_image.save("data/img.PNG")
        valid_pil_image.save("data/img.tiff")
        valid_pil_image.save("data/img.tif")
        valid_pil_image.save("data/img.bmp")

        text_filenames = [
            "data/img_tab.txt",
            "data/img_space.txt",
            "data/img_comma.txt",
            "data/img_pipe.txt",
            "data/img_semicolon.txt",
        ]

        # Put text data in a fake HTTP server
        for filename in text_filenames:
            with open(filename, "r") as fh:
                response_data = fh.read()  # type: str
                httpserver.expect_request(f"/{filename}").respond_with_data(
                    response_data, content_type="text/plain"
                )

        binary_filenames = [
            ("data/img.fits", "text/plain"),  # TODO: Change this type
            ("data/img_multiple.fits", "text/plain"),  # TODO: Change this type
            ("data/img_multiple.FITS", "text/plain"),  # TODO: Change this type
            ("data/img.npy", "text/plain"),  # TODO: Change this type
            ("data/img.jpg", "image/jpeg"),
            ("data/img.jpeg", "image/jpeg"),
            ("data/img.png", "image/png"),
            ("data/img.PNG", "image/png"),
            ("data/img.tif", "image/tiff"),
            ("data/img.tiff", "image/tiff"),
            ("data/img.bmp", "image/bmp"),
        ]

        # Put binary data in a fake HTTP server
        for filename, content_type in binary_filenames:
            with open(filename, "rb") as fh:
                response_data = fh.read()  # type: str
                httpserver.expect_request(f"/{filename}").respond_with_data(
                    response_data, content_type=content_type
                )

        # Extract an url (e.g. 'http://localhost:59226/)
        url = httpserver.url_for("")  # type: str

        # Extract the hostname (e.g. 'localhost:59226')
        hostname = re.findall("http://(.*)/", url)[0]  # type: str

        yield hostname

    finally:
        os.chdir(current_folder)


@pytest.mark.parametrize(
    "filename, exp_error, exp_message",
    [
        ("dummy", ValueError, "Image format not supported"),
        ("dummy.foo", ValueError, "Image format not supported"),
    ],
)
def test_invalid_filename(filename, exp_error, exp_message):
    """Test invalid filenames."""
    with pytest.raises(exp_error, match=exp_message):
        _ = load_image(filename)


@pytest.mark.parametrize(
    "filename, exp_data",
    [
        # FITS files
        ("data/img.fits", np.array([[1, 2], [3, 4]], np.uint16)),
        ("data/img.FITS", np.array([[1, 2], [3, 4]], np.uint16)),
        ("data/img_multiple.fits", np.array([[5, 6], [7, 8]], np.uint16)),
        ("data/img_multiple.FITS", np.array([[5, 6], [7, 8]], np.uint16)),
        (Path("data/img.fits"), np.array([[1, 2], [3, 4]], np.uint16)),
        (Path("./data/img.fits"), np.array([[1, 2], [3, 4]], np.uint16)),
        (Path("data/img_multiple.fits"), np.array([[5, 6], [7, 8]], np.uint16)),
        ("http://{host}/data/img.fits", np.array([[1, 2], [3, 4]], np.uint16)),
        ("http://{host}/data/img_multiple.fits", np.array([[5, 6], [7, 8]], np.uint16)),
        ("http://{host}/data/img_multiple.FITS", np.array([[5, 6], [7, 8]], np.uint16)),
        # Numpy binary files
        ("data/img.npy", np.array([[1, 2], [3, 4]], np.uint16)),
        (Path("data/img.npy"), np.array([[1, 2], [3, 4]], np.uint16)),
        ("http://{host}/data/img.npy", np.array([[1, 2], [3, 4]], np.uint16)),
        # Numpy text files
        ("data/img_tab.txt", np.array([[1, 2], [3, 4]], np.uint16)),
        ("data/img_space.txt", np.array([[1, 2], [3, 4]], np.uint16)),
        ("data/img_comma.txt", np.array([[1, 2], [3, 4]], np.uint16)),
        ("data/img_pipe.txt", np.array([[1, 2], [3, 4]], np.uint16)),
        ("data/img_semicolon.txt", np.array([[1, 2], [3, 4]], np.uint16)),
        (Path("data/img_tab.txt"), np.array([[1, 2], [3, 4]], np.uint16)),
        (Path("data/img_space.txt"), np.array([[1, 2], [3, 4]], np.uint16)),
        (Path("data/img_comma.txt"), np.array([[1, 2], [3, 4]], np.uint16)),
        (Path("data/img_pipe.txt"), np.array([[1, 2], [3, 4]], np.uint16)),
        (Path("data/img_semicolon.txt"), np.array([[1, 2], [3, 4]], np.uint16)),
        ("http://{host}/data/img_tab.txt", np.array([[1, 2], [3, 4]], np.uint16)),
        ("http://{host}/data/img_space.txt", np.array([[1, 2], [3, 4]], np.uint16)),
        ("http://{host}/data/img_comma.txt", np.array([[1, 2], [3, 4]], np.uint16)),
        ("http://{host}/data/img_pipe.txt", np.array([[1, 2], [3, 4]], np.uint16)),
        ("http://{host}/data/img_semicolon.txt", np.array([[1, 2], [3, 4]], np.uint16)),
        # JPG files
        ("data/img.jpg", np.array([[13, 19], [28, 34]])),
        ("data/img.jpeg", np.array([[13, 19], [28, 34]])),
        ("http://{host}/data/img.jpg", np.array([[13, 19], [28, 34]])),
        ("http://{host}/data/img.jpeg", np.array([[13, 19], [28, 34]])),
        # PNG files
        ("data/img.png", np.array([[10, 20], [30, 40]])),
        ("data/img.PNG", np.array([[10, 20], [30, 40]])),
        ("http://{host}/data/img.png", np.array([[10, 20], [30, 40]])),
        ("http://{host}/data/img.PNG", np.array([[10, 20], [30, 40]])),
        # TIFF files
        ("data/img.tif", np.array([[10, 20], [30, 40]])),
        ("data/img.tiff", np.array([[10, 20], [30, 40]])),
        ("http://{host}/data/img.tif", np.array([[10, 20], [30, 40]])),
        ("http://{host}/data/img.tiff", np.array([[10, 20], [30, 40]])),
        # BMP files
        ("data/img.bmp", np.array([[10, 20], [30, 40]])),
    ],
)
def test_load_image(
    valid_data2d_http_hostname: str, filename: t.Union[str, Path], exp_data: np.ndarray
):
    """Test function 'load_image' with local and remote files."""
    # Build a full url
    if isinstance(filename, Path):
        # Load data
        data_2d = load_image(filename)

    else:
        full_url = filename.format(host=valid_data2d_http_hostname)  # type: str

        # Load data
        data_2d = load_image(full_url)

    # Check 'data_2d
    np.testing.assert_equal(data_2d, exp_data)


def test_load_table_invalid_filename():
    with pytest.raises(FileNotFoundError):
        _ = load_table("dummy")


@pytest.mark.parametrize("filename", ["dummy.foo"])
def test_load_table_invalid_format(tmp_path: Path, filename: str):
    # Create an empty file
    full_filename = tmp_path.joinpath(filename)  # type: Path
    full_filename.touch()

    with pytest.raises(ValueError):
        _ = load_table(full_filename)


@pytest.mark.parametrize(
    "filename", ["valid_filename.txt", "valid_filename.TXT", "valid_filename.data"]
)
@pytest.mark.parametrize("delimiter", ["\t", " ", ",", "|", ";"])
def test_load_table_txtdata(tmp_path: Path, filename: str, delimiter: str, valid_table):
    full_filename = tmp_path.joinpath(filename)
    valid_table.to_csv(full_filename, header=None, index=None, sep=delimiter)

    assert full_filename.exists()

    table = load_table(full_filename)

    pd.testing.assert_frame_equal(table, valid_table)


@pytest.mark.skip(reason="Fix this test !")
@pytest.mark.parametrize("filename", ["valid_filename.xlsx"])
def test_load_table_xlsx(tmp_path: Path, filename: str, valid_table):
    full_filename = tmp_path.joinpath(filename)
    valid_table.to_excel(full_filename, header=False, index=False)

    assert full_filename.exists()

    table = load_table(full_filename)

    pd.testing.assert_frame_equal(table, valid_table)


@pytest.mark.parametrize("filename", ["valid_filename.csv", "valid_filename.CSV"])
@pytest.mark.parametrize("delimiter", ["\t", " ", ",", "|", ";"])
def test_load_table_csv(tmp_path: Path, filename: str, valid_table, delimiter: str):
    full_filename = tmp_path.joinpath(filename)
    valid_table.to_csv(full_filename, header=None, index=None, sep=delimiter)

    assert full_filename.exists()

    table = load_table(full_filename)

    pd.testing.assert_frame_equal(table, valid_table)

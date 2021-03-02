#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from PIL import Image
from pytest_httpserver import HTTPServer  # pip install pytest-httpserver

from pyxel.inputs_outputs import load_image, load_table


@pytest.fixture
def valid_hdus() -> fits.HDUList:
    """Create a valid HDUList with only one 'PrimaryHDU'."""
    # Create a new image
    data_2d = np.array([[1, 2], [3, 4]], dtype=np.uint16)

    hdu = fits.PrimaryHDU(data_2d)
    hdu_lst = fits.HDUList([hdu])

    return hdu_lst


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
def valid_data2d_folder(tmp_path: Path, httpserver: HTTPServer) -> Path:
    """Create a valid 2d files locally."""
    # Get current folder
    current_folder = Path().cwd()  # type: Path

    try:
        os.chdir(tmp_path)

        # Create folder 'data'
        Path("data").mkdir(parents=True, exist_ok=True)

        data_2d = np.array([[1, 2], [3, 4]], dtype=np.uint16)

        # Create a new FITS file based on 'filename'
        fits.writeto("data/frame2d.fits", data=data_2d)
        np.save("data/frame2d.npy", arr=data_2d)
        np.savetxt("data/frame2d_tab.txt", X=data_2d, delimiter="\t")
        np.savetxt("data/frame2d_space.txt", X=data_2d, delimiter=" ")
        np.savetxt("data/frame2d_comma.txt", X=data_2d, delimiter=",")
        np.savetxt("data/frame2d_pipe.txt", X=data_2d, delimiter="|")
        np.savetxt("data/frame2d_semicolon.txt", X=data_2d, delimiter=";")

        yield tmp_path

    finally:
        os.chdir(current_folder)


@pytest.fixture
def valid_data2d_http_hostname(
    valid_data2d_folder: Path, httpserver: HTTPServer
) -> str:
    """Create valid 2D files on a temporary HTTP server."""

    text_filenames = [
        "data/frame2d_tab.txt",
        "data/frame2d_space.txt",
        "data/frame2d_comma.txt",
        "data/frame2d_pipe.txt",
        "data/frame2d_semicolon.txt",
    ]

    # Put text data in a fake HTTP server
    for filename in text_filenames:
        with open(filename, "r") as fh:
            response_data = fh.read()  # type: str
            httpserver.expect_request(f"/{filename}").respond_with_data(
                response_data, content_type="text/plain"
            )

    binary_filenames = [
        "data/frame2d.fits",
        "data/frame2d.npy",
    ]

    # Put binary data in a fake HTTP server
    for filename in binary_filenames:
        with open(filename, "rb") as fh:
            response_data = fh.read()  # type: str
            httpserver.expect_request(f"/{filename}").respond_with_data(
                response_data, content_type="text/plain"
            )

    # Extract an url (e.g. 'http://localhost:59226/)
    url = httpserver.url_for("")  # type: str

    # Extract the hostname (e.g. 'localhost:59226')
    hostname = re.findall("http://(.*)/", url)[0]  # type: str

    yield hostname


@pytest.mark.parametrize(
    "filename, exp_error, exp_message",
    [
        ("dummy", ValueError, "Image format not supported"),
    ],
)
def test_invalid_filename(filename, exp_error, exp_message):
    """Test invalid filenames."""
    with pytest.raises(exp_error, match=exp_message):
        _ = load_image(filename)


@pytest.mark.parametrize("filename", ["dummy.foo"])
def test_invalid_format(tmp_path: Path, filename: str):
    # Create an empty file
    full_filename = tmp_path.joinpath(filename)  # type: Path
    full_filename.touch()

    with pytest.raises(ValueError):
        _ = load_image(full_filename)


@pytest.mark.parametrize(
    "filename",
    [
        # Local FITS files
        "data/frame2d.fits",
        "data/frame2d.FITS",
        Path("data/frame2d.fits"),
        Path("./data/frame2d.fits"),
        # Local Numpy binary files
        "data/frame2d.npy",
        Path("data/frame2d.npy"),
        # Local Numpy text files
        "data/frame2d_tab.txt",
        "data/frame2d_space.txt",
        "data/frame2d_comma.txt",
        "data/frame2d_pipe.txt",
        "data/frame2d_semicolon.txt",
        Path("data/frame2d_tab.txt"),
        Path("data/frame2d_space.txt"),
        Path("data/frame2d_comma.txt"),
        Path("data/frame2d_pipe.txt"),
        Path("data/frame2d_semicolon.txt"),
    ],
)
def test_load_image_local(valid_data2d_folder: Path, filename: str):
    """Check with a valid FITS file with a single 'PrimaryHDU'."""
    # Load data
    data_2d = load_image(filename)

    # Check 'data_2d
    np.testing.assert_equal(data_2d, np.array([[1, 2], [3, 4]], dtype=np.uint16))


@pytest.mark.parametrize(
    "url",
    [
        "http://{hostname}/data/frame2d.fits",
        "http://{hostname}/data/frame2d.npy",
        "http://{hostname}/data/frame2d_tab.txt",
        "http://{hostname}/data/frame2d_space.txt",
        "http://{hostname}/data/frame2d_comma.txt",
        "http://{hostname}/data/frame2d_pipe.txt",
        "http://{hostname}/data/frame2d_semicolon.txt",
    ],
)
def test_load_image_remote(valid_data2d_http_hostname: str, url: str):
    """Load a remote image."""
    # Build a full url
    full_url = url.format(hostname=valid_data2d_http_hostname)  # type: str

    # Load data
    data_2d = load_image(full_url)

    # Check 'data_2d
    np.testing.assert_equal(data_2d, np.array([[1, 2], [3, 4]], dtype=np.uint16))


@pytest.mark.parametrize(
    "filename",
    [
        "valid_frame.fits",
        "valid_frame.FITS",
        # "valid_frame.fits.gz"
    ],
)
def test_with_fits_multiple(
    tmp_path: Path, valid_multiple_hdus: fits.HDUList, filename: str
):
    """Check with a valid FITS file with a 'PrimaryHDU' and 'ImageHDU'."""
    # Create a new FITS file based on 'filename' and 'valid_hdus'
    full_filename = tmp_path.joinpath(filename)  # type: Path
    valid_multiple_hdus.writeto(full_filename)

    assert full_filename.exists()

    # Load FITS file
    data_2d = load_image(full_filename)

    # Check 'data_2d
    np.testing.assert_equal(data_2d, np.array([[5, 6], [7, 8]], dtype=np.uint16))


@pytest.mark.parametrize(
    "filename", ["valid_filename.txt", "valid_filename.TXT", "valid_filename.data"]
)
@pytest.mark.parametrize(
    "content",
    [
        pytest.param("1.1\t2.2\n3.3\t4.4", id="with tab"),
        pytest.param("1.1 2.2\n3.3 4.4", id="with space"),
        pytest.param("1.1,2.2\n3.3,4.4", id="with comma"),
        pytest.param("1.1|2.2\n3.3|4.4", id="with vertical-colon"),
        pytest.param("1.1;2.2\n3.3;4.4", id="with semicolon"),
    ],
)
def test_with_txt(tmp_path: Path, filename: str, content: str):
    # Create a new txt file based on 'filename'
    full_filename = tmp_path.joinpath(filename)  # type: Path
    with full_filename.open("w") as fh:
        fh.write(content)

    assert full_filename.exists()

    # Load TXT file
    data_2d = load_image(full_filename)

    # Check 'data_2d
    np.testing.assert_equal(data_2d, np.array([[1.1, 2.2], [3.3, 4.4]]))


@pytest.mark.parametrize(
    "filename",
    [
        "valid_filename.jpg",
        "valid_filename.jpeg",
        "valid_filename.png",
        "valid_filename.PNG",
        "valid_filename.tiff",
        "valid_filename.bmp",
    ],
)
def test_with_pil(tmp_path: Path, valid_pil_image: Image.Image, filename: str):
    """Check with a RGB uploaded image."""
    full_filename = tmp_path.joinpath(filename)
    valid_pil_image.save(full_filename)

    assert full_filename.exists()

    # Load image
    data_2d = load_image(full_filename)

    suffix = full_filename.suffix.lower()

    if suffix.startswith(".jpg") or suffix.startswith(".jpeg"):
        # Check compressed (lossy) jpg data_2d
        np.testing.assert_equal(data_2d, np.array([[13, 19], [28, 34]]))
    else:
        # Check data_2d
        np.testing.assert_equal(data_2d, np.array([[10, 20], [30, 40]]))


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

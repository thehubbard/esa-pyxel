#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
"""Pyxel examples downloader."""

import shutil
from os import listdir, remove, rmdir
from os.path import isdir, join
from zipfile import ZipFile

import requests


def download_examples(foldername: str = "pyxel-examples", force: bool = False) -> None:
    """Download and save examples from Pyxel Data Gitlab repository in the working directory.

    Parameters
    ----------
    foldername: str
    force: bool

    Returns
    -------
    None
    """

    if isdir(foldername) and not force:
        raise OSError(
            f"Folder {foldername} already exists. Either delete it, use a different name, or use the force argument."
        )
    elif isdir(foldername) and force:
        shutil.rmtree(foldername)

    url = "https://gitlab.com/esa/pyxel-data/-/archive/master/pyxel-data-master.zip"
    response = requests.get(url)

    with open("examples_tmp.zip", "wb") as tmp:
        tmp.write(response.content)

    with ZipFile("examples_tmp.zip", "r") as zipobj:
        zipobj.extractall(foldername)

    root = foldername
    for filename in listdir(join(root, "pyxel-data-master")):
        shutil.move(join(root, "pyxel-data-master", filename), join(root, filename))
    rmdir(join(root, "pyxel-data-master"))

    remove("examples_tmp.zip")

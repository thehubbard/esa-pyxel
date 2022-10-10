#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Module to read/write ASDF files."""

import typing as t
from pathlib import Path

import asdf


def to_asdf(filename: t.Union[str, Path], dct: t.Mapping[str, t.Any]) -> None:
    """Write data to a ASDF file."""
    if dct["version"] != 1:
        raise NotImplementedError

    # TODO: Remove this
    # TODO: Use .to_numpy() ?
    dct["data"]["charge"]["frame"] = dct["data"]["charge"]["frame"].to_dict(
        orient="list"
    )

    af = asdf.AsdfFile(dct)
    af.write_to(filename)
    print("Hello World")


def from_asdf(filename: t.Union[str, Path]) -> t.Mapping[str, t.Any]:
    """Read data from a HDF5 file."""
    dct = {}
    with asdf.open(filename, copy_arrays=True) as af:

        # TODO: Use a JSON schema to validate 'dct'
        if "version" not in af or "type" not in af:
            raise ValueError("Missing 'version' and/or 'type' !")

        version = af["version"]  # type: int
        if version != 1:
            raise NotImplementedError

        dct["version"] = version
        dct["type"] = af["type"]

        # Get properties
        dct["properties"] = af["properties"]
        dct["data"] = af["data"]

        return dct

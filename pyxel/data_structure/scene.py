#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Pyxel Scene class to track multi-wavelength photon."""

import typing as t

if t.TYPE_CHECKING:
    from scopesim import Source


class Scene:
    """Scene class defining and storing information of all multi-wavelength photon."""

    def __init__(self, source: "Source"):
        self._source = source  # type: Source

    def __eq__(self, other) -> bool:
        raise NotImplementedError

    @property
    def data(self) -> "Source":
        """Get a multi-wavelength object."""
        return self._source

    def to_dict(self) -> t.Mapping:
        """Convert an instance of `Scene` to a `dict`."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, dct: t.Mapping) -> "Scene":
        """Create a new instance of a `Scene` object from a `dict`."""
        raise NotImplementedError

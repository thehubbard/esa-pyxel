#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Pyxel Scene class to track multi-wavelength photon."""

import typing as t

if t.TYPE_CHECKING:
    import numpy as np
    from astropy.io.fits import ImageHDU
    from astropy.table import Table
    from scopesim import Source


class Scene:
    """Scene class defining and storing information of all multi-wavelength photon."""

    def __init__(self, source: "Source"):
        self._source = source  # type: Source

    def __eq__(self, other) -> bool:
        return type(self) == type(other) and self.data == other.data

    @property
    def data(self) -> "Source":
        """Get a multi-wavelength object."""
        return self._source

    def to_dict(self) -> t.Mapping:
        """Convert an instance of `Scene` to a `dict`."""
        meta = self._source.meta  # type: t.Mapping
        table_fields = self._source.table_fields  # type: t.Sequence[Table]
        image_fields = self._source.image_fields  # type: t.Sequence[ImageHDU]

        # Create 'tables'
        tables = [
            {
                "data": table.to_pandas(),
                "units": {
                    key.replace("_unit", ""): value
                    for key, value in table.meta.items()
                    if key.endswith("_unit")
                },
            }
            for table in table_fields
        ]  # type: t.Sequence[t.Mapping]

        images = [
            {"header": dict(image.header), "data": np.asarray(image.data)}
            for image in image_fields
        ]  # type: t.Sequence[t.Mapping]

        return {"meta": meta, "tables": tables, "images": images}

    @classmethod
    def from_dict(cls, dct: t.Mapping) -> "Scene":
        """Create a new instance of a `Scene` object from a `dict`."""
        from astropy.io.fits import Header, ImageHDU
        from astropy.table import Table
        from scopesim import Source

        meta = dct["meta"]  # type: t.Mapping
        tables = dct["tables"]  # type: t.Mapping
        images = dct["images"]  # type: t.Mapping

        table_fields = [
            Table.from_pandas(dataframe=table["data"], units=table["units"])
            for table in tables
        ]  # type: t.Sequence[Table]

        image_fields = [
            ImageHDU(data=img["data"], header=Header(img["header"])) for img in images
        ]  # type: t.Sequence[ImageHDU]

        src = Source(
            meta=meta,
            image_fields=image_fields,
            table_fields=table_fields,
        )  # type: Source

        return cls(src)

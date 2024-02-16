#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from collections.abc import Mapping
from enum import Enum, auto
from typing import Optional, Union

import astropy.units as u
import numpy as np
import pytest
import xarray as xr

from pyxel.data_structure import Photon
from pyxel.detectors import (
    CCD,
    CCDGeometry,
    Characteristics,
    Environment,
    WavelengthHandling,
)
from pyxel.models.photon_collection import optical_psf
from pyxel.models.photon_collection.poppy import (
    CircularAperture,
    HexagonAperture,
    MultiHexagonalAperture,
    RectangleAperture,
    SecondaryObscuration,
    SineWaveWFE,
    SquareAperture,
    ThinLens,
    ZernikeWFE,
    create_optical_parameter,
)

_ = pytest.importorskip("poppy")


@pytest.fixture
def ccd_3x3() -> CCD:
    """Create a valid CCD detector."""
    detector = CCD(
        geometry=CCDGeometry(
            row=3,
            col=3,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
            pixel_scale=1.65,
        ),
        environment=Environment(wavelength=600.0),
        characteristics=Characteristics(),
    )
    detector.photon.array = np.zeros(detector.geometry.shape, dtype=float)
    return detector


@pytest.fixture
def ccd_100x100_no_photon() -> CCD:
    """Create a valid CCD detector."""
    detector = CCD(
        geometry=CCDGeometry(
            row=100,
            col=100,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
            # pixel_scale=1.65,
        ),
        environment=Environment(),
        characteristics=Characteristics(),
    )

    detector.set_readout(times=[1.0], start_time=0.0)

    return detector


@pytest.mark.parametrize(
    "dct, exp_parameter",
    [
        pytest.param(
            {}, None, marks=pytest.mark.xfail(raises=KeyError, strict=True), id="Empty"
        ),
        pytest.param(
            {"item": "CircularAperture", "radius": 3.14},
            CircularAperture(radius=3.14 * u.m),
            id="CircularAperture",
        ),
        pytest.param(
            {
                "item": "ThinLens",
                "nwaves": 1.1,
                "radius": 2.2,
                "reference_wavelength": 600.0,
            },
            ThinLens(nwaves=1.1, radius=2.2 * u.m, reference_wavelength=600.0 * u.nm),
            id="ThinLens",
        ),
        pytest.param(
            {"item": "SquareAperture", "size": 1},
            SquareAperture(size=1 * u.m),
            id="SquareAperture",
        ),
        pytest.param(
            {"item": "RectangularAperture", "width": 1.1, "height": 2.2},
            RectangleAperture(width=1.1 * u.m, height=2.2 * u.m),
            id="RectangularAperture",
        ),
        pytest.param(
            {"item": "HexagonAperture", "side": 1.1},
            HexagonAperture(side=1.1 * u.m),
            id="HexagonAperture",
        ),
        pytest.param(
            {"item": "MultiHexagonalAperture", "side": 1.1, "rings": 2, "gap": 3.3},
            MultiHexagonalAperture(side=1.1 * u.m, rings=2, gap=3.3 * u.m),
            id="MultiHexagonalAperture",
        ),
        pytest.param(
            {
                "item": "SecondaryObscuration",
                "secondary_radius": 1.1,
                "n_supports": 2,
                "support_width": 3.3,
            },
            SecondaryObscuration(
                secondary_radius=1.1 * u.m, n_supports=2, support_width=3.3 * u.m
            ),
            id="SecondaryObscuration",
        ),
        pytest.param(
            {
                "item": "ZernikeWFE",
                "radius": 1.1,
                "coefficients": [2.2, 3.3],
                "aperture_stop": 4.4,
            },
            ZernikeWFE(radius=1.1 * u.m, coefficients=[2.2, 3.3], aperture_stop=4.4),
            id="ZernikeWFE",
        ),
        pytest.param(
            {
                "item": "SineWaveWFE",
                "spatialfreq": 2.2,
                "amplitude": 3.3,
                "rotation": 4.4,
            },
            SineWaveWFE(
                spatialfreq=2.2 * (1 / u.m), amplitude=3.3 * u.um, rotation=4.4
            ),
            id="SineWaveWFE",
        ),
    ],
)
def test_create_optical_parameter(dct: Mapping, ccd_3x3: CCD, exp_parameter):
    """Test function 'create_optical_parameter'."""
    parameter = create_optical_parameter(
        dct, selected_wavelength=ccd_3x3.environment.wavelength * u.nm
    )

    assert parameter == exp_parameter


class PhotonType(Enum):
    """Used only for testing."""

    NoPhoton = auto()
    Photon_2D = auto()
    Photon_3D = auto()


@pytest.mark.parametrize(
    "photon_type, env_wavelength, wavelength",
    [
        pytest.param(
            PhotonType.Photon_2D, None, 620.0, id="Mono. with wavelength (float)"
        ),
        pytest.param(PhotonType.Photon_2D, None, 620, id="Mono. with wavelength (int)"),
        pytest.param(PhotonType.Photon_2D, 620.0, None, id="Mono. without wavelength"),
        pytest.param(
            PhotonType.Photon_3D,
            None,
            (620.0, 680.0),
            id="Multi with wavelength (float)",
        ),
        pytest.param(
            PhotonType.Photon_3D, None, (620, 680), id="Multi with wavelength (int)"
        ),
        pytest.param(
            PhotonType.Photon_3D, None, (600.0, 700.0), id="Multi with wavelength2"
        ),
        pytest.param(
            PhotonType.Photon_3D,
            WavelengthHandling(cut_on=620.0, cut_off=680.0, resolution=2),
            None,
            id="Multi without wavelength",
        ),
    ],
)
@pytest.mark.parametrize(
    "geo_pixel_scale, pixelscale",
    [
        pytest.param(None, 1.65, id="With pixelscale"),
        pytest.param(1.65, None, id="Without pixelscale"),
    ],
)
@pytest.mark.parametrize(
    "optical_system",
    [pytest.param([{"item": "CircularAperture", "radius": 1.0}], id="circle")],
)
def test_optical_psf(
    photon_type: PhotonType,
    env_wavelength: Union[WavelengthHandling, float, None],
    wavelength: Union[float, tuple[float, float], None],
    geo_pixel_scale: Optional[float],
    pixelscale: Optional[float],
    optical_system,
    ccd_100x100_no_photon: CCD,
):
    """Test function 'optical_psf'."""
    detector = ccd_100x100_no_photon
    detector.environment._wavelength = env_wavelength
    detector.geometry._pixel_scale = geo_pixel_scale

    # Check if 'photon' is empty
    assert detector.photon == Photon(geo=detector.geometry)
    assert detector.photon.ndim == 0

    # Add Photons (or not)
    if photon_type is PhotonType.NoPhoton:
        # Do nothing
        pass
    elif photon_type is PhotonType.Photon_2D:
        detector.photon.array = np.zeros(
            shape=(detector.geometry.row, detector.geometry.col),
            dtype=float,
        )
    else:
        detector.photon.array_3d = xr.DataArray(
            np.zeros(
                shape=(3, detector.geometry.row, detector.geometry.col),
                dtype=float,
            ),
            dims=["wavelength", "y", "x"],
            coords={"wavelength": [620.0, 640.0, 680.0]},
        )

    optical_psf(
        detector=detector,
        fov_arcsec=5,
        optical_system=optical_system,
        wavelength=wavelength,
        pixelscale=pixelscale,
        apply_jitter=False,
        jitter_sigma=0.007,
    )

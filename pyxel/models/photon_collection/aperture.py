#   Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#  #
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this Pyxel package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.

"""Aperture model."""

import astropy.units as u
import numpy as np
import xarray as xr
from astropy import wcs
from astropy.coordinates import SkyCoord

from pyxel.data_structure import Scene
from pyxel.detectors import Detector


def extract_wavelength(
    scene: Scene,
    wavelength: float,
) -> xr.Dataset:
    """Extracts xarray Dataset of Scene for selected wavelength.

    Parameters
    ----------
    scene : Scene
        Pyxel scene object.
    wavelength : float
        Selected wavelength. Unit: nm.

    Returns
    -------
    selected_data : xr.Dataset
    """

    assert len(scene.data["/list"]) == 1
    data: xr.Dataset = scene.data["/list/0"].to_dataset()

    # get dataset with x, y, weight and flux of scene for selected wavelength.
    selected_data: xr.Dataset = data.sel(wavelength=wavelength, method="nearest")

    return selected_data


def convert_flux(
    flux: xr.DataArray,
    t_exp: float,
    aperture: float,
) -> np.ndarray:
    """Convert flux (photon/s/cm2) to photon/s/pixel.

    Parameters
    ----------
    flux : np.ndarray
        Scene object. Unit: photon/pixel/s/cm2.
    t_exp : float
        Exposure time. Unit: s.
    aperture : float
        Collecting area of the telescope. Unit: m.

    Returns
    -------
    np.ndarray
        Converted flux in photon/s/pixel.
    """

    col_area = np.pi * (aperture * 1e2 / 2) ** 2
    flux_converted = flux * t_exp * col_area

    return flux_converted


# def project_stars(
#
# ):


def simple_aperture(
    detector: Detector,
    pixel_scale: float,
    aperture: float,
    wavelength: float,
):
    """Convert scene(photon/s/cm2) to photon.

    Parameters
    ----------
    detector : Detector
        Pyxel detector object.
    pixel_scale : float
        Pixel scale. Unit: arcsec/pixel.
    aperture : float
        Collecting area of the telescope. Unit: m.
    wavelength : float
        Wavelength. Unit: nm.
    """
    # define variables
    pixel_scale = pixel_scale * u.arcsec / u.pixel
    geo = detector.geometry
    rows = geo.row
    cols = geo.col

    # get dataset for given wavelength and scene object.
    selected_data = extract_wavelength(scene=detector.scene, wavelength=wavelength)

    # get flux in ph/s/cm^2
    flux = selected_data.flux.values * u.nm * u.Unit(selected_data["flux"].units)
    # get time in s
    time = detector.time_step * u.s

    # get flux converted to photon
    converted_flux = convert_flux(flux=flux, t_exp=time, aperture=aperture)

    # extra function
    # we project the stars in the FOV:

    # array of star coordinates
    stars_coords = SkyCoord(
        selected_data["x"].values * u.arcsec,
        selected_data["y"].values * u.arcsec,
        frame="icrs",
    )

    # TODO: get this from input from the "load_star_map" model
    # coordinates of telescope pointing
    telescope_ra = "56.75"
    telescope_dec = "24.1167"
    coords_detector = SkyCoord(ra=telescope_ra, dec=telescope_dec, unit="degree")

    # magic
    cdelt = (np.array([-1.0, 1.0]) * pixel_scale).to(u.deg / u.pixel)
    # shape of detector 100,100
    crpix = np.array([geo.row.shape[0] / 2, geo.col.shape[1] / 2]) * u.pixel
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = crpix
    w.wcs.crval = [coords_detector.ra.deg, coords_detector.dec.deg]
    w.wcs.cdelt = cdelt
    w.wcs.crota = [0, -0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # or like scopesim:
    # https://github.com/AstarVienna/ScopeSim/blob/dev_master/scopesim/optics/image_plane_utils.py#L698
    # gives different result.

    # da = cdelt[0]
    # db = cdelt[1]
    # x0 = crpix[0]
    # y0 = crpix[1]
    # a0 = selected_data["x"].values * u.arcsec
    # b0 = selected_data["y"].values * u.arcsec
    # a = float(telescope_ra) * u.deg
    # b = float(telescope_dec) * u.deg

    # convert stars coordinate to detector coordinates
    # detector_coords_x = x0 + 1. / da * (a - a0)
    # detector_coords_y = y0 + 1. / db * (b - b0)

    detector_coords_x = np.round(
        w.world_to_pixel_values(stars_coords.ra, stars_coords.dec)[0]
    )

    detector_coords_y = np.round(
        w.world_to_pixel_values(stars_coords.ra, stars_coords.dec)[1]
    )

    selected_data["detector_coords_x"] = xr.DataArray(
        detector_coords_x, dims="ref", attrs={"units": "pixel"}
    )
    selected_data["detector_coords_y"] = xr.DataArray(
        detector_coords_y, dims="ref", attrs={"units": "pixel"}
    )

    # make sure that only stars inside the detector
    selected_data_new = (
        selected_data.query(ref="detector_coords_x > 0")
        .query(ref=f"detector_coords_x < {cols}")
        .query(ref=f"detector_coords_y < {rows}")
        .query(ref="detector_coords_y > 0")
    )

    # shape of the detector
    projection = np.zeros([geo.row.shape[1], geo.col.shape[2]])

    # TODO: check if x, y or y, x.
    #  From Pixel Coordinate Conventions:
    #  Following Numpy indexing, the y coordinate corresponds to rows and the x coordinate corresponds to columns.

    for x, group_x in selected_data_new.groupby("detector_coords_x"):
        for y, group_y in group_x.groupby("detector_coords_y"):
            projection[int(x), int(y)] = group_y["flux"].values.sum()

    # extra func over

    detector.photon.array = projection

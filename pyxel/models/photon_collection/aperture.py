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
    wavelength_band: tuple[float, float],
) -> xr.Dataset:
    """Extract xarray Dataset of Scene for selected wavelength band.

    Parameters
    ----------
    scene : Scene
        Pyxel scene object.
    wavelength_band : float
        Selected wavelength band. Unit: nm.

    Returns
    -------
    selected_data : xr.Dataset
    """

    assert len(scene.data["/list"]) == 1
    data: xr.Dataset = scene.data["/list/0"].to_dataset()

    start_wavelength, end_wavelength = wavelength_band

    # get dataset with x, y, weight and flux of scene for selected wavelength band.
    selected_data: xr.Dataset = data.sel(
        wavelength=slice(start_wavelength, end_wavelength)
    )

    return selected_data


def integrate_flux(
    flux: xr.DataArray,
) -> xr.DataArray:
    # integrate flux
    integrated_flux = flux.integrate(coord="wavelength")
    integrated_flux.attrs["units"] = str(u.Unit(flux.units) * u.nm)

    return integrated_flux


def convert_flux(
    flux: u.Quantity,
    t_exp: float,
    aperture: float,
) -> u.Quantity:
    """Convert flux (photon/s/cm2) to photon/s/pixel.

    Parameters
    ----------
    flux : u.Quantity
        Flux. Unit: photon/pixel/s/cm2.
    t_exp : float
        Exposure time. Unit: s.
    aperture : float
        Collecting area of the telescope. Unit: m.

    Returns
    -------
    u.Quantity
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
    wavelength_band: tuple[float, float],
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
    wavelength_band : tuple[float, float]
        Wavelength band. Unit: nm.
    """
    # define variables
    pixel_scale = pixel_scale * u.arcsec / u.pixel
    geo = detector.geometry
    rows = geo.row
    cols = geo.col

    # get dataset for given wavelength and scene object.
    selected_data: xr.Dataset = extract_wavelength(
        scene=detector.scene, wavelength_band=wavelength_band
    )

    # integrate flux
    integrated_flux: xr.DataArray = integrate_flux(flux=selected_data["flux"])

    # get flux in ph/s/cm^2
    flux = u.Quantity(integrated_flux, unit=integrated_flux.units)
    # get time in s
    time = detector.time_step * u.s

    # get flux converted to ph
    converted_flux: u.Quantity = convert_flux(flux=flux, t_exp=time, aperture=aperture)

    # load converted flux to selected dataset
    selected_data["converted_flux"] = xr.DataArray(
        converted_flux, dims="ref", attrs={"units": str(converted_flux.unit)}
    )

    # TODO: split up to an extra function
    # we project the stars in the FOV:

    # array of star coordinates really needed?
    stars_coords = SkyCoord(
        selected_data["x"].values * u.arcsec,
        selected_data["y"].values * u.arcsec,
        frame="icrs",
    )

    # coordinates of telescope pointing
    telescope_ra: u.Quantity = (selected_data["x"].values * u.arcsec).mean()
    telescope_dec: u.Quantity = (selected_data["y"].values * u.arcsec).mean()
    coords_detector = SkyCoord(ra=telescope_ra, dec=telescope_dec, unit="degree")

    # using world coordinate system to convert to pixel
    # TODO: add info from https://heasarc.gsfc.nasa.gov/docs/fcg/standard_dict.html
    cdelt = (np.array([-1.0, 1.0]) * pixel_scale).to(u.deg / u.pixel)
    crpix = np.array([geo.row / 2, geo.col / 2]) * u.pixel
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
    selected_data_new = selected_data.copy(deep=True)
    selected_data_query = (
        selected_data_new.query(ref="detector_coords_x > 0")
        .query(ref=f"detector_coords_x < {cols}")
        .query(ref=f"detector_coords_y < {rows}")
        .query(ref="detector_coords_y > 0")
    )

    # convert to int
    selected_data2 = selected_data_query.copy(deep=True)

    selected_data2["detector_coords_x"] = selected_data2["detector_coords_x"].astype(
        int
    )
    selected_data2["detector_coords_y"] = selected_data2["detector_coords_y"].astype(
        int
    )

    # shape of the detector
    projection = np.zeros([geo.row, geo.col])

    for x, group_x in selected_data2.groupby("detector_coords_x"):
        for y, group_y in group_x.groupby("detector_coords_y"):
            projection[int(y), int(x)] = group_y["converted_flux"].values.sum()

    # extra func over

    detector.photon.array = projection

#   Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#  #
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this Pyxel package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.

"""Scene generator creates Scopesim Source object."""
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
import requests
import scopesim
from astropy.table import Table
from astroquery.gaia import Gaia
from specutils import Spectrum1D
from synphot import SourceSpectrum

from pyxel.data_structure import Scene
from pyxel.detectors import Detector

if TYPE_CHECKING:
    from astropy.io.votable import tree


def retrieve_objects_from_gaia(
    right_ascension: float,
    declination: float,
    fov_radius: float,
) -> tuple[Table, dict[int, Table]]:
    """Retrieve objects from GAIA Catalog for giver coordinates and FOV.

    Parameters
    ----------
        right_ascension: float
            RA coordinate in degree.
        declination: float
            DEC coordinate in degree.
        fov_radius: float
            FOV radius of telescope optics.

    Returns
    -------
        Table, Sequence of Tables

    Raises
    ------
        ConnectionError
            If the connection to the GAIA database cannot be established.

    Examples
    --------
        >>> positions, spectra = retrieve_objects_from_gaia(
        ...     right_ascension=56.75,
        ...     declination=24.1167,
        ...     fov_radius=0.05,
        ... )
        >>> positions
            source_id             ra                dec         has_xp_sampled phot_bp_mean_mag phot_g_mean_mag phot_rp_mean_mag
                                 deg                deg                              mag              mag             mag
        ----------------- ------------------ ------------------ -------------- ---------------- --------------- ----------------
        66727234683960320 56.760485086776846 24.149991010998228           True        14.734505       14.433147        13.954827
        65214031805717376     56.74561610052 24.089174782613686           True        12.338661       11.940813        11.368548
        65225851555715328 56.726951308177455 24.111718134110838           True        14.627676        13.91212        13.091035
        65226195153096192 56.736700233543914 24.149504345515066           True        14.272486       13.613182        12.804853
        >>> len(spectra)
        4
        >>> spectra[66727234683960320]
        wavelength      flux       flux_error
        nm      W / (nm m2)   W / (nm m2)
    ---------- ------------- -------------
         336.0 4.1858373e-17 5.8010027e-18
         338.0  4.101217e-17  4.343636e-18
         340.0  3.499973e-17 3.4906054e-18
         342.0 3.0911544e-17 3.0178371e-18
           ...           ...           ...
        1014.0  1.742315e-17  4.146798e-18
        1016.0 1.5590336e-17  4.358966e-18
        1018.0 1.3888942e-17 4.2265315e-18
        1020.0  1.344579e-17 4.1775913e-18
    """
    # Unlimited rows.
    Gaia.ROW_LIMIT = -1
    # we get the data from GAIA DR3
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    try:
        # Query for the catalog to search area with coordinates in FOV of optics.
        job = Gaia.launch_job_async(
            f"SELECT source_id, ra, dec, has_xp_sampled, phot_bp_mean_mag, phot_g_mean_mag, phot_rp_mean_mag \
        FROM gaiadr3.gaia_source \
        WHERE CONTAINS(POINT('ICRS', ra, dec),CIRCLE('ICRS',{right_ascension},{declination},{fov_radius}))=1 \
        AND has_xp_sampled = 'True'"
        )

        # get the results from the query job
        result: Table = job.get_results()

        # set parameters to load data from Gaia catalog
        retrieval_type = "XP_SAMPLED"
        data_release = "Gaia DR3"
        data_structure = "COMBINED"

        # load spectra from stars
        spectra_dct: dict[str, list[tree.Table]] = Gaia.load_data(
            ids=result["source_id"],
            retrieval_type=retrieval_type,
            data_release=data_release,
            data_structure=data_structure,
        )
    except requests.HTTPError as exc:
        raise ConnectionError(
            "Error when trying to communicate with the Gaia database"
        ) from exc
    # save key
    key = f"{retrieval_type}_{data_structure}.xml"

    spectra: dict[int, Table] = {
        spectrum.get_field_by_id("source_id").value: spectrum.to_table()
        for spectrum in spectra_dct[key]
    }

    return result, spectra


def load_objects_from_gaia(
    right_ascension: float,
    declination: float,
    fov_radius: float,
) -> scopesim.Source:
    """Load objects from GAIA Catalog for given coordinates and FOV.

    Parameters
    ----------
    right_ascension: float
        RA coordinate in degree.
    declination: float
        DEC coordinate in degree.
    fov_radius: float
        FOV radius of telescope optics.

    Returns
    -------
    scopesim.Source
        Scopesim Source object in the FOV at given coordinates found by the GAIA catalog.
    """
    positions, spectra = retrieve_objects_from_gaia(
        right_ascension=right_ascension,
        declination=declination,
        fov_radius=fov_radius,
    )

    # loop through all sources and create Source Spectrum to attach to a list of spectra
    spec_list: list[SourceSpectrum] = []
    for source_id in positions["source_id"]:
        spectrum: Table = spectra[source_id]

        # create 1D spectrum
        spec = Spectrum1D(
            spectral_axis=spectrum["wavelength"].quantity,
            flux=spectrum["flux"].quantity,
        )
        # turn 1D spectrum into synphot Source spectrum
        source_spectrum = SourceSpectrum.from_spectrum1d(spec)
        spec_list.append(source_spectrum)

    # Create astropy.table with relevant input parameters

    # Convert the positions to arcseconds and center around 0.0, 0.0.
    # The centering is necessary because ScopeSIM cannot actually 'point' the telescope.
    x = 3600 * (positions["ra"].quantity - positions["ra"].quantity.mean())
    y = 3600 * (positions["dec"].quantity - positions["dec"].quantity.mean())

    # set index as ref.
    ref = np.arange(len(positions), dtype=int)

    # select magnitude band. Could be an input parameter of model.
    weight: u.Quantity = positions["phot_bp_mean_mag"].quantity

    tbl = Table(
        names=["x", "y", "ref", "weight"],
        data=[x.to(u.arcsec), y.to(u.arcsec), ref, weight],
    )

    source = scopesim.Source(table=tbl, spectra=spec_list)
    return source


def generate_scene(
    detector: Detector,
    right_ascension: float,
    declination: float,
    fov_radius: float,
):
    """Generate scene from scopesim Source object loading stars from the GAIA catalog.

    Parameters
    ----------
    detector : Detector
        Pyxel Detector object.
    right_ascension: float
        RA coordinate in degree.
    declination: float
        DEC coordinate in degree.
    fov_radius: float
        FOV radius of telescope optics.

    """
    source: scopesim.Source = load_objects_from_gaia(
        right_ascension=right_ascension, declination=declination, fov_radius=fov_radius
    )

    detector.scene = Scene(source)

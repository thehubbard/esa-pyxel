#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Model to calculate the QE based on the characteristics of the detector (Front/Back Illumination, Charge Collection Efficiency, Absorption coefficient, Reflectivity, Epilayer thickness, Poly gate thickness)."""

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd
import xarray as xr
from astropy.units import Quantity

if TYPE_CHECKING:
    from pyxel.detectors import Detector


def exponential_qe(
    detector: "Detector",
    filename: Union[str, Path],
    x_epi: float,
    detector_type: str,  # BI: Back-Illuminated, FI: Front-Illuminated
    default_wavelength: Union[str, float] = None,  # User must provide a value
    x_poly: float = 0.0,  # Default x_poly to 0.0, change only if the detector is front-illuminated
    delta_t: float = 0.0,  # Temperature difference from 300 K
    cce: float = 1.0,  # Default Charge Collection Efficiency (CCE)
) -> None:
    """
    Apply QE with temperature correction for absorptivity using a provided or backup coefficient `c`.

    Parameters
    ----------
    detector : Detector
        Pyxel Detector object.
    filename : str or Path
        Path to the CSV file containing reflectivity, absorptivity, and c.
    x_poly : float
        Thickness of the poly layer in cm.
    x_epi : float
        Thickness of the epitaxial layer in cm.
    delta_t : float
        Temperature difference from 300 K (default: 0.0).
    detector_type : str
        Type of detector ("BI" for Back-Illuminated, "FI" for Front-Illuminated).
    cce : float
        Charge Collection Efficiency (default: 1.0).
    default_wavelength : str or float
        Wavelength in nm for 2D photon arrays, or 'multi' for multiple wavelengths (no default value).
    """
    # Validate delta_t to ensure resulting temperature is 300 K
    resulting_temperature = detector.environment.temperature - delta_t
    if resulting_temperature != 300.0:
        raise ValueError(
            f"The temperature provided (delta_t = {delta_t}) does not match with the environment. "
            f"Temperature difference: {resulting_temperature} K."
        )

    # Ensure default_wavelength is provided
    if default_wavelength is None:
        raise ValueError(
            "You must specify a `default_wavelength` value in nm or use 'multi' for multiple wavelengths."
        )

    # Define valid wavelength range for the equation
    valid_wavelength_range = (250.0, 1400.0)  # Range in nm

    # Validate default_wavelength for single-wavelength mode
    if isinstance(default_wavelength, (int, float)):
        if not (valid_wavelength_range[0] <= default_wavelength <= valid_wavelength_range[1]):
            raise ValueError(
                f"Wavelength {default_wavelength} nm is out of the valid range "
                f"{valid_wavelength_range} for the equation."
            )

    # Validate detector_type
    if detector_type not in ["BI", "FI"]:
        raise ValueError(
            "Invalid detector type. Choose 'BI' (Back-Illuminated) or 'FI' (Front-Illuminated)."
        )

    # Enforce x_poly = 0 for Back-Illuminated detectors
    if detector_type == "BI" and x_poly != 0.0:
        print(
            "Warning: x_poly should be 0 for Back-Illuminated detectors. Setting x_poly to 0."
        )
        x_poly = 0.0

    # Validate non-negative thickness values
    if x_epi < 0 or x_poly < 0:
        raise ValueError(
            "Epitaxial thickness (x_epi) and poly layer thickness (x_poly) must be non-negative."
        )

    # Step 1: Detect the shape of the photon array
    photon_array = detector.photon.array

    if len(photon_array.shape) == 2:  # If 2D
        if default_wavelength == "multi":
            raise ValueError(
                "Photon array is 2D, but you specified 'multi' for `default_wavelength`. Ensure the photon array matches your wavelength input."
            )
        elif isinstance(default_wavelength, (int, float)):
            print(
                "Photon array is 2D. Transforming it into a 3D array with a single wavelength slice."
            )
            dummy_wavelength = np.array([default_wavelength])  # Single wavelength value
            # Generate coordinates for x and y
            y_coords = np.arange(photon_array.shape[0])  # Row indices
            x_coords = np.arange(photon_array.shape[1])  # Column indices

            # Create 3D xarray DataArray
            photon_array_3d = xr.DataArray(
                np.expand_dims(photon_array, axis=0),  # Add wavelength as a new dimension
                coords={"wavelength": dummy_wavelength, "y": y_coords, "x": x_coords},
                dims=["wavelength", "y", "x"],
            )
            detector.photon.array_3d = photon_array_3d  # Assign back to the detector
        else:
            raise ValueError(
                "Invalid `default_wavelength` value. Must be a numeric value in nm or 'multi' for multiple wavelengths."
            )

    elif len(photon_array.shape) == 3:  # If already 3D
        if default_wavelength != "multi":
            print(
                "Photon array is 3D, but `default_wavelength` is not 'multi'. Proceeding with the existing wavelength data."
            )
        print("Photon array is 3D. Proceeding normally.")
        photon_array_3d = xr.DataArray(
            photon_array,
            coords={
                "wavelength": np.arange(
                    photon_array.shape[0]
                ),  # Generate wavelength coordinates
                "y": np.arange(photon_array.shape[1]),
                "x": np.arange(photon_array.shape[2]),
            },
            dims=["wavelength", "y", "x"],
        )
        detector.photon.array_3d = photon_array_3d  # Ensure it's an xarray.DataArray
    else:
        raise ValueError(
            f"Unexpected photon array dimensions: {photon_array.shape}. Expected 2D or 3D."
        )

    # Convert x_epi to Quantity
    x_epi_cm = Quantity(x_epi, unit="cm")

    # Read total detector thickness from the detector object
    total_thickness = Quantity(detector.geometry.total_thickness, unit="um")
    if x_epi_cm > total_thickness.to("cm"):
        raise ValueError(
            f"x_epi ({x_epi_cm}) cannot be greater than the total detector thickness ({total_thickness.to('cm')})."
        )

    # Load data from the provided CSV file
    qe_data = pd.read_csv(filename)

    # Check for required columns
    required_columns = {"reflectivity", "absorptivity", "wavelength"}
    if not required_columns.issubset(qe_data.columns):
        raise ValueError(f"CSV file must contain the columns: {required_columns}")

    # Extract data
    reflectivity = qe_data["reflectivity"].values
    absorptivity = Quantity(qe_data["absorptivity"].values, unit="1/cm")
    wavelength = Quantity(qe_data["wavelength"].values, unit="nm")

    # Embedded full-range `c` values provided by the developers
    embedded_c_values = Quantity(
        [
            -9.00e-05,
            -1.50e-04,
            -3.10e-04,
            -3.30e-04,
            8.00e-05,
            2.50e-04,
            3.20e-04,
            1.50e-04,
            7.00e-05,
            3.00e-05,
            0.00e00,
            -1.40e-04,
            4.20e-04,
            9.10e-04,
            2.60e-03,
            3.30e-03,
            3.10e-03,
            2.90e-03,
            2.90e-03,
            2.80e-03,
            2.80e-03,
            2.90e-03,
            2.90e-03,
            3.00e-03,
            3.00e-03,
            3.10e-03,
            3.10e-03,
            3.20e-03,
            3.30e-03,
            3.30e-03,
            3.30e-03,
            3.40e-03,
            3.40e-03,
            3.40e-03,
            3.40e-03,
            3.40e-03,
            3.50e-03,
            3.50e-03,
            3.50e-03,
            3.50e-03,
            3.50e-03,
            3.50e-03,
            3.60e-03,
            3.60e-03,
            3.60e-03,
            3.70e-03,
            3.70e-03,
            3.70e-03,
            3.70e-03,
            3.70e-03,
            3.70e-03,
            3.70e-03,
            3.70e-03,
            3.70e-03,
            3.80e-03,
            4.00e-03,
            4.10e-03,
            4.20e-03,
            4.40e-03,
            4.50e-03,
            4.60e-03,
            4.70e-03,
            4.90e-03,
            5.10e-03,
            5.20e-03,
            5.40e-03,
            5.60e-03,
            5.70e-03,
            5.90e-03,
            6.20e-03,
            6.50e-03,
            6.90e-03,
            7.30e-03,
            7.80e-03,
            8.30e-03,
            9.00e-03,
            9.70e-03,
            1.05e-02,
            1.12e-02,
            1.20e-02,
            1.35e-02,
            1.45e-02,
            1.55e-02,
            1.60e-02,
            1.65e-02,
            1.75e-02,
            1.80e-02,
            1.85e-02,
            1.90e-02,
            2.00e-02,
            2.10e-02,
            2.30e-02,
            2.60e-02,
            3.20e-02,
            3.45e-02,
            3.55e-02,
            3.80e-02,
            3.90e-02,
            4.05e-02,
            4.10e-02,
            4.30e-02,
            4.40e-02,
            4.55e-02,
            4.70e-02,
            5.00e-02,
            5.25e-02,
            5.50e-02,
            5.80e-02,
            6.10e-02,
            6.50e-02,
            6.70e-02,
            6.75e-02,
            6.80e-02,
            6.85e-02,
            6.90e-02,
            7.00e-02,
            7.10e-02,
            7.20e-02,
            7.30e-02,
            7.40e-02,
            7.50e-02,
        ],
        unit="1/K",
    )  # Embedded c-values

    embedded_wavelengths = Quantity(
        np.arange(250, 1410, 10), unit="nm"
    )  # Wavelengths for embedded `c` values

    # Check if the 'c' column exists, otherwise use embedded values
    if "c" in qe_data.columns:
        c_values = Quantity(qe_data["c"].values, unit="1/K")
    else:
        c_values = (
            np.interp(
                x=wavelength.value,  # Use the numeric values of wavelength
                xp=embedded_wavelengths.value,  # Use the numeric values of embedded wavelengths
                fp=embedded_c_values.value,  # Use the numeric values of embedded c-values
            )
            * embedded_c_values.unit
        )  # Add the unit back to the interpolated result

    # Correct absorptivity for temperature, if delta_t != 0
    if delta_t != 0:
        delta_t = Quantity(delta_t, unit="K")
        absorptivity = absorptivity * np.exp(c_values * delta_t)

    # Calculate absorption length (L_A) as the inverse of corrected absorptivity
    l_a = 1 / absorptivity

    # Define the QE formula based on the detector type
    if detector_type == "BI":
        qe = cce * (1 - reflectivity) * (1 - np.exp(-x_epi_cm / l_a))
    elif detector_type == "FI":
        qe = (
            cce
            * (1 - reflectivity)
            * np.exp(-x_poly / l_a)
            * (1 - np.exp(-x_epi_cm / l_a))
        )
    else:
        raise ValueError("Invalid detector type. Choose 'BI' or 'FI'.")

    # Create an xarray Dataset for QE, aligned to wavelength
    qe_dataset = xr.Dataset(
        {"QE": (["wavelength"], qe)},
        coords={"wavelength": wavelength},
    )

    # Step 8: Apply QE to photon array and integrate charge
    qe_dataset = xr.Dataset(
        {"QE": (["wavelength"], qe)}, coords={"wavelength": wavelength}
    )
    qe_interpolated = qe_dataset.interp(wavelength=photon_array_3d["wavelength"])
    charge_array = photon_array_3d * qe_interpolated["QE"]
    # Check the number of wavelengths in charge_array
    if len(charge_array["wavelength"]) == 1:
        # If only one wavelength, squeeze the wavelength dimension
        charges = charge_array.squeeze(dim="wavelength")
        print("Single wavelength detected. Skipping integration over wavelength.")
    else:
        # Otherwise, integrate over the wavelength
        charges = charge_array.integrate(coord="wavelength")
        print("Multiple wavelengths detected. Performing integration over wavelength.")

    # Add charges to the detector
    detector.charge.add_charge_array(np.asarray(charges))
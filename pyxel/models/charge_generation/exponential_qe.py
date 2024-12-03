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
import math
from astropy.units import Quantity

if TYPE_CHECKING:
    from pyxel.detectors import Detector


def exponential_qe(
    detector: "Detector",
    filename: Union[str, Path],
    x_epi: float,
    detector_type: str,  # BI: Back-Illuminated, FI: Front-Illuminated
    default_wavelength: Union[str, float, None] = None,  # User must provide a value
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
    # Validate if resulting_temperature is close to 300 K
    if not math.isclose(resulting_temperature, 300, abs_tol=0.01):
        raise ValueError(
            f"The temperature provided does not match with the environment."
        )
    # Ensure default_wavelength is provided
    if default_wavelength is None:
        raise ValueError(
            "You must specify a `default_wavelength` value in nm or use 'multi' for multiple wavelengths."
        )

    # Define valid wavelength range for the equation
    valid_wavelength_range = (250.0, 1450.0)  # Range in nm

    # Validate default_wavelength for single-wavelength mode
    if isinstance(default_wavelength, (int, float)):
        if not (
            valid_wavelength_range[0] <= default_wavelength <= valid_wavelength_range[1]
        ):
            raise ValueError(
                f"Wavelength is out of the valid range "
                f"for the equation."
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
    if detector.photon.ndim == 2:
        photon_2d: np.ndarray = detector.photon.array_2d

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
            y_coords = np.arange(photon_2d.shape[0])  # Row indices
            x_coords = np.arange(photon_2d.shape[1])  # Column indices

            # Create 3D xarray DataArray
            photon_array_3d = xr.DataArray(
                np.expand_dims(photon_2d, axis=0),  # Add wavelength as a new dimension
                coords={"wavelength": dummy_wavelength, "y": y_coords, "x": x_coords},
                dims=["wavelength", "y", "x"],
            )
        else:
            raise ValueError(
                "Invalid `default_wavelength` value. Must be a numeric value in nm or 'multi' for multiple wavelengths."
            )

    elif detector.photon.ndim == 3:
        if default_wavelength != "multi":
            print(
                "Photon array is 3D, but `default_wavelength` is not 'multi'. Proceeding with the existing wavelength data."
            )
        print("Photon array is 3D. Proceeding normally.")
        photon_array_3d = detector.photon.array_3d

    else:
        raise ValueError(
            f"Unexpected photon array dimensions: {detector.photon.shape}. Expected 2D or 3D."
        )

    # Convert x_epi to Quantity
    x_epi_cm = Quantity(x_epi, unit="cm")
    x_poly_cm = Quantity(x_poly, unit="cm")

    # Read total detector thickness from the detector object
    total_thickness = Quantity(detector.geometry.total_thickness, unit="um")
    if x_epi_cm > total_thickness.to("cm"):
        raise ValueError(
            f"x_epi cannot be greater than the total detector thickness."
        )

    # Load data from the provided CSV file
    qe_data = pd.read_csv(filename)

    # Check for required columns
    required_columns = ["reflectivity", "absorptivity", "wavelength"]
    if not set(required_columns).issubset(qe_data.columns):
        raise ValueError(f"CSV file must contain the columns: {', '.join(required_columns)}")

    # Validate that no NaN values exist in the required columns
    nan_columns = [
        col for col in required_columns if qe_data[col].isna().any()
    ]
    if nan_columns:
        raise ValueError(
            f"NaN values found in the file. All values for 'wavelength', 'reflectivity', and 'absorptivity' must be present."
        )

    # Validate that no negative values exist in the required columns
    negative_columns = [
        col for col in required_columns if (qe_data[col] < 0).any()
    ]
    if negative_columns:
        raise ValueError(
            f"Negative values found in the following columns: {', '.join(negative_columns)}. "
            f"All values for 'wavelength', 'reflectivity', and 'absorptivity' must be non-negative."
        )

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

    embedded_absorptivity_values = Quantity(
        [
            1840219.313, 1970020.255, 2180032.591, 2370107.258, 2290112.714,
            1770182.741, 1460131.192, 1299833.96, 1180096.44, 1099927.028,
            1040188.716, 999402.2122, 972244.3148, 951255.5361, 933230.7961,
            917315.0366, 903100.905, 889457.0984, 876379.5276, 863889.645,
            851993.563, 840692.9717, 829985.7003, 819866.4986, 810328.7327,
            801364.4082, 792965.8568, 785125.8917, 777837.6556, 771094.5403,
            764890.0175, 759217.6618, 754071.1094, 749444.0822, 745330.3542,
            741723.7431, 738618.1025, 735907.318, 733585.3345, 731646.1347,
            730083.7368, 728892.1945, 728065.5961, 727598.0633, 727483.7523,
            727716.8536, 728291.5938, 729202.2343, 730443.0705, 732008.4298,
            733892.6722, 736090.1897, 738595.4062, 741402.7728, 744506.7655,
            747901.8855, 751582.6596, 755543.6376, 759779.3916, 764284.5143,
            769053.6188, 774081.3373, 779362.321, 784891.2403, 790662.7842,
            796671.6608, 802912.5966, 809380.3363, 816069.6422, 822975.2969,
            830092.1027, 837414.881, 844938.4712, 852657.7305, 860567.5335,
            868662.7726, 876938.3586, 885389.2205, 894010.3058, 902796.5802,
            911743.0283, 920844.6536, 930096.4789, 939493.5457, 949030.9153,
            958703.6686, 968506.9068, 978435.7513, 988485.3433, 998650.8447,
            1008927.436, 1019320.319, 1029814.712, 1040415.854, 1051129.003,
            1061960.435, 1072916.443, 1084003.342, 1095227.466, 1106595.161,
            1118112.791, 1129786.735, 1141623.387, 1153629.152, 1165810.447,
            1178173.701, 1190725.354, 1203461.856
        ],
        unit="1/cm",
    )

    embedded_wavelengths = Quantity(
        np.arange(250, 1460, 10), unit="nm"
    )  # Wavelengths for embedded `c` values

    # Check if the 'c' column exists, otherwise use embedded values
    if "c" in qe_data.columns:
        # Check for NaN values in the 'c' column
        if qe_data["c"].isna().any():
            raise ValueError("NaN values found in the 'c' column. All values must be present.")
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
        absorptivity = (
                np.interp(
                    x=wavelength.value,  # Use the numeric values of wavelength
                    xp=embedded_wavelengths.value,  # Use the numeric values of embedded wavelengths
                    fp=embedded_absorptivity_values.value,  # Use the numeric values of embedded c-values
                )
                * embedded_c_values.unit
        )  # Add the unit back to the interpolated result

    # Correct absorptivity for temperature, if delta_t != 0
    if delta_t != 0:
        delta_t = Quantity(delta_t, unit="K")
        absorptivity = absorptivity * np.exp(c_values * delta_t)

    # Define the QE formula based on the detector type
    if detector_type == "BI":
        qe = cce * (1 - reflectivity) * (1 - np.exp(-x_epi_cm * absorptivity))
    elif detector_type == "FI":
        qe = (
            cce
            * (1 - reflectivity)
            * np.exp(-x_poly_cm * absorptivity)
            * (1 - np.exp(-x_epi_cm * absorptivity))
        )
    else:
        raise ValueError("Invalid detector type. Choose 'BI' or 'FI'.")

    # Create an xarray Dataset for QE, aligned to wavelength
    qe_dataset = xr.Dataset(
        {"QE": (["wavelength"], qe)},
        coords={"wavelength": wavelength},
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

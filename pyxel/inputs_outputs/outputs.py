#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
"""Classes for creating outputs."""
import logging
import typing as t
from glob import glob
from pathlib import Path
from time import strftime

import h5py as h5
import numpy as np
import pandas as pd
import xarray as xr
from astropy.io import fits as fits
from typing_extensions import Literal

from pyxel import __version__ as version

if t.TYPE_CHECKING:
    from pyxel.detectors import Detector
    from pyxel.pipelines import Processor

    class SaveToFile(t.Protocol):
        """TBW."""

        def __call__(
            self, data: t.Any, name: str, with_auto_suffix: bool = True
        ) -> Path:
            """TBW."""
            ...


ValidName = Literal[
    "detector.image.array", "detector.signal.array", "detector.pixel.array"
]
ValidFormat = Literal["fits", "hdf", "npy", "txt", "csv", "png"]


class Outputs:
    """TBW."""

    def __init__(
        self,
        output_folder: t.Union[str, Path],
        save_data_to_file: t.Optional[
            t.Sequence[t.Mapping[ValidName, t.Sequence[ValidFormat]]]
        ] = None,
    ):
        self._log = logging.getLogger(__name__)

        self.output_dir = create_output_directory(output_folder)  # type: Path

        # TODO: Not related to a plot. Use by 'single' and 'parametric' modes.
        self.save_data_to_file = (
            save_data_to_file
        )  # type: t.Optional[t.Sequence[t.Mapping[str, t.Sequence[str]]]]

    def __repr__(self):
        cls_name = self.__class__.__name__  # type: str
        return f"{cls_name}<output_dir={self.output_dir!r}>"

    def save_to_fits(
        self, data: np.ndarray, name: str, with_auto_suffix: bool = True
    ) -> Path:
        """Write array to FITS file."""
        name = str(name).replace(".", "_")

        if with_auto_suffix:
            filename = apply_run_number(self.output_dir.joinpath(f"{name}_??.fits"))
        else:
            filename = self.output_dir / f"{name}.fits"

        full_filename = filename.resolve()  # type: Path
        self._log.info("Save to FITS - filename: '%s'", full_filename)

        hdu = fits.PrimaryHDU(data)
        hdu.header["PYXEL_V"] = (str(version), "Pyxel version")
        hdu.writeto(full_filename, overwrite=False, output_verify="exception")

        return full_filename

    def save_to_hdf(
        self, data: "Detector", name: str, with_auto_suffix: bool = True
    ) -> Path:
        """Write detector object to HDF5 file."""
        name = str(name).replace(".", "_")

        if with_auto_suffix:
            filename = apply_run_number(self.output_dir.joinpath(f"{name}_??.h5"))
        else:
            filename = self.output_dir / f"{name}.h5"

        full_filename = filename.resolve()  # type: Path

        with h5.File(full_filename, "w") as h5file:
            h5file.attrs["pyxel-version"] = str(version)
            if name == "detector":
                detector_grp = h5file.create_group("detector")
                for array, name in zip(
                    [
                        data.signal.array,
                        data.image.array,
                        data.photon.array,
                        data.pixel.array,
                        data.charge.frame,
                    ],
                    ["Signal", "Image", "Photon", "Pixel", "Charge"],
                ):
                    dataset = detector_grp.create_dataset(name, shape=np.shape(array))
                    dataset[:] = array
            else:
                raise NotImplementedError
                # detector_grp = h5file.create_group("data")
                # dataset = detector_grp.create_dataset(name, shape=np.shape(data))
                # dataset[:] = data
        return filename

    def save_to_txt(
        self, data: np.ndarray, name: str, with_auto_suffix: bool = True
    ) -> Path:
        """Write data to txt file."""
        name = str(name).replace(".", "_")

        if with_auto_suffix:
            filename = apply_run_number(self.output_dir.joinpath(f"{name}_??.txt"))
        else:
            filename = self.output_dir / f"{name}.txt"

        full_filename = filename.resolve()  # type: Path
        np.savetxt(full_filename, data, delimiter=" | ", fmt="%.8e")

        return full_filename

    def save_to_csv(
        self, data: pd.DataFrame, name: str, with_auto_suffix: bool = True
    ) -> Path:
        """Write Pandas Dataframe or Numpy array to a CSV file."""
        name = str(name).replace(".", "_")

        if with_auto_suffix:
            filename = apply_run_number(self.output_dir.joinpath(f"{name}_??.csv"))
        else:
            filename = self.output_dir / f"{name}.csv"

        full_filename = filename.resolve()
        try:
            data.to_csv(full_filename, float_format="%g")
        except AttributeError:
            np.savetxt(full_filename, data, delimiter=",", fmt="%.8e")

        return full_filename

    def save_to_npy(
        self, data: np.ndarray, name: str, with_auto_suffix: bool = True
    ) -> Path:
        """Write Numpy array to Numpy binary npy file."""
        name = str(name).replace(".", "_")

        if with_auto_suffix:
            filename = apply_run_number(self.output_dir.joinpath(f"{name}_??.npy"))
        else:
            filename = self.output_dir / f"{name}.npy"

        full_filename = filename.resolve()  # type: Path

        np.save(file=full_filename, arr=data)
        return full_filename

    def save_to_file(
        self,
        processor: "Processor",
        prefix: t.Optional[str] = None,
        with_auto_suffix: bool = True,
    ) -> t.Sequence[Path]:
        """Save outputs into file(s).

        Parameters
        ----------
        processor : Processor

        Returns
        -------
        list of ``Path``
            TBW.
        """
        save_methods = {
            "fits": self.save_to_fits,
            "hdf": self.save_to_hdf,
            "npy": self.save_to_npy,
            "txt": self.save_to_txt,
            "csv": self.save_to_csv,
            # "png": self.save_to_png,
        }  # type: t.Mapping[ValidFormat, SaveToFile]

        filenames = []  # type: t.List[Path]

        dct: t.Mapping[ValidName, t.Sequence[ValidFormat]]
        if self.save_data_to_file:
            for dct in self.save_data_to_file:
                # TODO: Why looking at first entry ? Check this !
                # Get first entry of `dict` 'item'
                first_item: t.Tuple[ValidName, t.Sequence[ValidFormat]]
                first_item, *_ = dct.items()

                obj: ValidName
                format_list: t.Sequence[ValidFormat]
                obj, format_list = first_item

                data = processor.get(obj)  # type: np.ndarray

                if prefix:
                    name = f"{prefix}_{obj}"  # type: str
                else:
                    name = obj

                out_format: ValidFormat
                for out_format in format_list:
                    func = save_methods[out_format]  # type: SaveToFile
                    filename = func(
                        data=data, name=name, with_auto_suffix=with_auto_suffix
                    )  # type: Path

                    filenames.append(filename)

        return filenames

    def save_to_netcdf(
        self, data: xr.Dataset, name: str, with_auto_suffix: bool = False
    ) -> Path:
        """Write Xarray dataset to NetCDF file.

        Parameters
        ----------
        data: xr.Dataset
        name: str

        Returns
        -------
        filename: path
        """
        name = str(name).replace(".", "_")
        filename = self.output_dir.joinpath(name + ".nc")
        data.to_netcdf(filename)
        return filename


# TODO: Create unit tests
# TODO: Refactor this in 'def apply_run_number(folder, template_filename) -> Path
def apply_run_number(template_filename: Path) -> Path:
    """Convert the file name numeric placeholder to a unique number.

    :param template_filename:
    :return:
    """
    path_str = str(template_filename)
    if "?" in path_str:
        # TODO: Use method 'Path.glob'
        dir_list = sorted(glob(path_str))

        p_0 = path_str.find("?")
        p_1 = path_str.rfind("?")
        template = path_str[slice(p_0, p_1 + 1)]
        path_str = path_str.replace(template, "{:0%dd}" % len(template))

        last_num = 0
        if len(dir_list):
            path_last = dir_list[-1]
            last_num = int(path_last[slice(p_0, p_1 + 1)])
        last_num += 1
        path_str = path_str.format(last_num)

    return Path(path_str)


# TODO: the log file should directly write in 'output_dir'
def save_log_file(output_dir: Path) -> None:
    """Move log file to the outputs directory of the simulation."""
    log_file = Path("pyxel.log").resolve(strict=True)  # type: Path

    new_log_filename = output_dir.joinpath(log_file.name)
    log_file.rename(new_log_filename)


def create_output_directory(output_folder: t.Union[str, Path]) -> Path:
    """Create output directory in the output folder.

    Parameters
    ----------
    output_folder: str or Path

    Returns
    -------
    output_dir: Path
    """

    add = ""
    count = 0

    while True:
        try:
            output_dir = (
                Path(output_folder)
                .joinpath("run_" + strftime("%Y%m%d_%H%M%S") + add)
                .resolve()
            )  # type: Path

            output_dir.mkdir(parents=True, exist_ok=False)

        except FileExistsError:
            count += 1
            add = "_" + str(count)
            continue

        else:
            return output_dir


# # TODO: Refactor this function
# def update_fits_header(
#     header: dict, key: t.Union[str, list, tuple], value: t.Any
# ) -> None:
#     """TBW.
#
#     Parameters
#     ----------
#     header
#     key
#     value
#     """
#     if isinstance(value, (str, int, float)):
#         result = value  # type: t.Union[str, int, float]
#     else:
#         result = repr(value)
#
#     if isinstance(result, str):
#         result = result[0:24]
#
#     if isinstance(key, (list, tuple)):
#         key = "/".join(key)
#
#     key = key.replace(".", "/")[0:36]
#     header[key] = value

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Utility functions for creating outputs."""
import logging
import typing as t
import warnings
from enum import Enum
from glob import glob
from numbers import Number
from pathlib import Path
from shutil import copy2
from time import strftime

import astropy.io.fits as fits
import attr
import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from pyxel import __version__ as version
from pyxel.calibration.util import ResultType

if t.TYPE_CHECKING:
    from ..pipelines import Processor
    from ..detectors import Detector
    from ..parametric.parametric import ParametricAnalysis
    from ..parametric.parameter_values import ParameterValues

    class SaveToFile(t.Protocol):
        """TBW."""

        def __call__(self, data: np.ndarray, name: str) -> Path:
            """TBW."""
            ...


@attr.s(auto_attribs=True, slots=True, frozen=True)
class Result:
    """TBW."""

    result: np.ndarray  # TODO: Use a `DataFrame` ?
    plot: np.ndarray


@attr.s(auto_attribs=True, slots=True, frozen=True)
class PlotArguments:
    """TBW."""

    title: t.Optional[str] = None
    xscale: str = "linear"
    yscale: str = "linear"
    xlabel: t.Optional[str] = None
    ylabel: t.Optional[str] = None
    xlim: t.Tuple[t.Optional[float], t.Optional[float]] = (None, None)
    ylim: t.Tuple[t.Optional[float], t.Optional[float]] = (None, None)
    xticks: t.Any = None
    yticks: t.Any = None
    sci_x: bool = False
    sci_y: bool = False
    grid: bool = False
    axis: t.Any = None
    bins: t.Optional[int] = None  # TODO: This should not be here !

    @classmethod
    def from_dict(cls, dct: t.Dict[str, t.Any]) -> "PlotArguments":
        """TBW."""
        return cls(**dct)

    def to_dict(self) -> dict:
        """TBW."""
        return attr.asdict(self)


@attr.s(auto_attribs=True, slots=True)
class ParametricPlot:
    """TBW."""

    x: str
    y: str
    plot_args: PlotArguments

    @classmethod
    def from_dict(cls, dct: dict) -> "ParametricPlot":
        """TBW."""
        return cls(x=dct["x"], y=dct["y"], plot_args=dct["plot_args"])


class PlotType(Enum):
    """TBW."""

    Histogram = "hist"
    Graph = "graph"
    Scatter = "scatter"


# TODO: Create a special Output class for 'parametric_plot', 'calibration_plot' and
#       'single_plot' ?
# TODO: Example
#       >>> class ParametricOutputs:
#       ...     def __init__(self, parametric, ...): ...
class Outputs:
    """TBW."""

    def __init__(
        self,
        output_folder: t.Union[str, Path],
        save_data_to_file: t.Optional[t.List[t.Dict[str, t.List[str]]]] = None,
        save_parameter_to_file: t.Optional[dict] = None,
        parametric_plot: t.Optional[dict] = None,  # TODO: See issue #80
        calibration_plot: t.Optional[t.Dict[str, t.Any]] = None,  # TODO: See issue #79
        single_plot: t.Optional[dict] = None,  # TODO: See issue #78
    ):
        """TBW."""
        self._log = logging.getLogger(__name__)

        # Check number of inputs
        num_inputs = sum(
            [
                parametric_plot is not None,
                calibration_plot is not None,
                single_plot is not None,
            ]
        )  # type: int
        if num_inputs not in (0, 1):
            raise ValueError(
                "Too much parameters. You should have only parameter "
                "'parametric_plot', 'calibration_plot' or 'single_plot'."
            )

        # self.input_file = None  # type: t.Optional[Path]

        # Parameter(s) specific for 'Parametric'
        self.parametric_plot = None  # type: t.Optional[ParametricPlot]
        self.parameter_keys = []  # type: t.List[str]

        if parametric_plot is not None:
            self.parametric_plot = ParametricPlot.from_dict(parametric_plot)

        # Parameter(s) specific for 'Calibration'
        self.calibration_plot = None  # type: t.Optional[t.Dict[str, t.Any]]
        if calibration_plot is not None:
            self.calibration_plot = calibration_plot

        # Parameter(s) specific for 'Single'
        self._single_plot = None  # type: t.Optional[dict]
        if single_plot is not None:
            self._single_plot = single_plot

        self.user_plt_args = None  # type: t.Optional[PlotArguments]
        self.save_parameter_to_file = save_parameter_to_file  # type: t.Optional[dict]
        self.output_dir = (
            Path(output_folder).joinpath("run_" + strftime("%Y%m%d_%H%M%S")).resolve()
        )  # type: Path

        # if save_data_to_file is None:
        #     self.save_data_to_file = [{'detector.image.array': ['fits']}]       # type: list
        # else:
        # TODO: Not related to a plot. Use by 'single' and 'parametric' modes.
        self.save_data_to_file = save_data_to_file or [
            {"detector.image.array": ["fits"]}
        ]  # type: t.List[t.Dict[str, t.List[str]]]

        if self.output_dir.exists():
            raise IsADirectoryError("Directory exists.")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # self.default_ax_args = {
        #     "xlabel": None,
        #     "ylabel": None,
        #     "title": None,
        #     "axis": None,
        #     "grid": False,
        #     "xscale": "linear",
        #     "yscale": "linear",
        #     "xticks": None,
        #     "yticks": None,
        #     "xlim": [None, None],
        #     "ylim": [None, None],
        #     "sci_x": False,
        #     "sci_y": False,
        # }  # type: dict
        self.default_ax_args = PlotArguments()

        self.default_plot_args = {
            "color": None,
            "marker": ".",
            "linestyle": "",
        }  # type: dict

        self.default_hist_args = {
            "bins": None,
            "range": None,
            "density": None,
            "log": False,
            "cumulative": False,
            "histtype": "step",
            "color": None,
            "facecolor": None,
        }  # type: dict
        self.default_scatter_args = {"size": None, "cbar_label": None}  # type: dict

        self.plt_args = {}  # type: dict

        # TODO: Create an object self._fig here ?
        fig, ax = plt.subplots(1, 1)

        self._fig = fig
        self._ax = ax

        plt.close(self._fig)

    def __repr__(self):
        """TBW."""
        cls_name = self.__class__.__name__  # type: str

        if self.parametric_plot is not None:
            mode = "parametric"
        elif self.calibration_plot is not None:
            mode = "calibration"
        else:
            mode = "single"

        return f"{cls_name}<mode={mode!r}, output_dir={self.output_dir!r}>"

    @property
    def fig(self) -> plt.Figure:
        """Get the current ``Figure``."""
        return self._fig

    # TODO: Rename this method to 'copy_config_file' ??
    def set_input_file(self, filename: t.Union[str, Path]) -> Path:
        """Copy a YAML configuration filename into its output directory.

        Parameters
        ----------
        filename : str or Path
            YAML filename to copy

        Returns
        -------
        Path
            Returns the copied YAML filename.
        """
        input_file = Path(filename)
        copy2(input_file, self.output_dir)

        # TODO: sort filenames ?
        copied_input_file_it = self.output_dir.glob("*.yaml")  # type: t.Iterator[Path]
        copied_input_file = next(copied_input_file_it)  # type: Path

        with copied_input_file.open("a") as file:
            file.write("\n#########")
            file.write(f"\n# Pyxel version: {version}")
            file.write("\n#########")

        return copied_input_file

    # TODO: the log file should directly write in 'output_dir'
    def save_log_file(self) -> None:
        """Move log file to the output directory of the simulation."""
        log_file = Path("pyxel.log").resolve(strict=True)  # type: Path

        new_log_filename = self.output_dir.joinpath(log_file.name)
        log_file.rename(new_log_filename)

    def new_file(self, filename: str) -> Path:
        """TBW."""
        new_filename = self.output_dir.joinpath(filename)  # type: Path
        new_filename.touch()

        return new_filename

    # TODO: Specific to 'single_plot' ?
    def save_to_png(self, data: np.ndarray, name: str) -> Path:
        """Write array to bitmap PNG image file.

        Parameters
        ----------
        data : array
        name : str

        Returns
        -------
        Path
            TBW.
        """
        row, col = data.shape
        name = str(name).replace(".", "_")
        filename = apply_run_number(
            self.output_dir.joinpath(f"{name}_??.png")
        )  # type: Path

        dpi = 300
        self._fig.set_size_inches(min(col / dpi, 10.0), min(row / dpi, 10.0))

        # ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        # ax.set_axis_off()
        #
        # fig.add_axes(ax)
        # plt.imshow(data, cmap="gray", extent=[0, col, 0, row])
        self._fig.savefig(filename, dpi=dpi)

        return filename

    def save_to_fits(self, data: np.ndarray, name: str) -> Path:
        """Write array to FITS file.

        Parameters
        ----------
        data
        name

        Returns
        -------
        Path
            TBW.
        """
        name = str(name).replace(".", "_")
        filename = apply_run_number(
            self.output_dir.joinpath(f"{name}_??.fits")
        ).resolve()  # type: Path

        self._log.info("Save to FITS - filename: '%s'", filename)

        hdu = fits.PrimaryHDU(data)
        hdu.header["PYXEL_V"] = (str(version), "Pyxel version")
        hdu.writeto(filename, overwrite=False, output_verify="exception")

        return filename

    def save_to_hdf(self, data: "Detector", name: str) -> Path:
        """Write detector object to HDF5 file."""
        name = str(name).replace(".", "_")
        filename = apply_run_number(self.output_dir.joinpath(f"{name}_??.h5"))
        with h5.File(filename, "w") as h5file:
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
                    dataset = detector_grp.create_dataset(name, np.shape(array))
                    dataset[:] = array
            else:
                detector_grp = h5file.create_group("data")
                dataset = detector_grp.create_dataset(name, np.shape(data))
                dataset[:] = data
        return filename

    def save_to_txt(self, data: np.ndarray, name: str) -> Path:
        """Write data to txt file."""
        name = str(name).replace(".", "_")
        filename = apply_run_number(self.output_dir.joinpath(name + "_??.txt"))
        np.savetxt(filename, data, delimiter=" | ", fmt="%.8e")
        return filename

    def save_to_csv(self, data: pd.DataFrame, name: str) -> Path:
        """Write Pandas Dataframe or Numpy array to a CSV file."""
        name = str(name).replace(".", "_")
        filename = apply_run_number(
            self.output_dir.joinpath(name + "_??.csv")
        )  # type: Path
        try:
            data.to_csv(filename, float_format="%g")
        except AttributeError:
            np.savetxt(filename, data, delimiter=",", fmt="%.8e")
        return filename

    def save_to_npy(self, data: np.ndarray, name: str) -> Path:
        """Write Numpy array to Numpy binary npy file."""
        name = str(name).replace(".", "_")
        filename = apply_run_number(self.output_dir.joinpath(name + "_??.npy"))
        np.save(file=filename, arr=data)
        return filename

    def save_plot(self, filename: str = "figure_??") -> Path:
        """Save plot figure in PNG format, close figure and create new canvas for next plot."""
        new_filename = self.output_dir.joinpath(filename + ".png")  # type: Path
        output_filename = apply_run_number(new_filename).resolve()  # type: Path

        self._log.info("Save plot in filename '%s'", output_filename)
        self._fig.savefig(output_filename)
        # plt.close('all')
        # plt.figure()

        return output_filename

    def plot_graph(
        self, x: np.ndarray, y: np.ndarray, args: t.Optional[dict] = None
    ) -> None:
        """TBW.

        Parameters
        ----------
        x
        y
        args
        """
        ax_args0, plt_args0 = self.update_args(plot_type=PlotType.Graph, new_args=args)

        user_plt_args_dct = None  # type: t.Optional[dict]
        if isinstance(self.user_plt_args, PlotArguments):
            user_plt_args_dct = self.user_plt_args.to_dict()

        ax_args, plt_args = self.update_args(
            plot_type=PlotType.Graph,
            new_args=user_plt_args_dct,
            ax_args=ax_args0,
            plt_args=plt_args0,
        )

        self._ax.plot(
            x,
            y,
            color=plt_args["color"],
            marker=plt_args["marker"],
            linestyle=plt_args["linestyle"],
        )
        update_plot(ax_args=ax_args, ax=self._ax)
        # plt.draw()

    def plot_histogram(self, data: np.ndarray, args: t.Optional[dict] = None) -> None:
        """TBW.

        Parameters
        ----------
        data
        args
        """
        assert self.user_plt_args is not None

        user_plt_args_dct = None  # type: t.Optional[dict]
        if isinstance(self.user_plt_args, PlotArguments):
            user_plt_args_dct = self.user_plt_args.to_dict()

        ax_args0, plt_args0 = self.update_args(
            plot_type=PlotType.Histogram, new_args=args
        )
        ax_args, plt_args = self.update_args(
            plot_type=PlotType.Histogram,
            new_args=user_plt_args_dct,
            ax_args=ax_args0,
            plt_args=plt_args0,
        )
        if isinstance(data, np.ndarray):
            data = data.flatten()

        self._ax.hist(
            x=data,
            bins=plt_args["bins"],
            range=plt_args["range"],
            density=plt_args["density"],
            log=plt_args["log"],
            cumulative=plt_args["cumulative"],
            histtype=plt_args["histtype"],
            color=plt_args["color"],
            facecolor=plt_args["facecolor"],
        )

        update_plot(ax_args=ax_args, ax=self._ax)

    def plot_scatter(
        self,
        x: np.ndarray,
        y: np.ndarray,
        color: t.Optional[str] = None,
        args: t.Optional[dict] = None,
    ) -> None:
        """TBW.

        Parameters
        ----------
        x
        y
        color
        args
        """
        user_plt_args_dct = None  # type: t.Optional[dict]

        if isinstance(self.user_plt_args, PlotArguments):
            user_plt_args_dct = self.user_plt_args.to_dict()

        ax_args0, plt_args0 = self.update_args(
            plot_type=PlotType.Scatter, new_args=args
        )
        ax_args, plt_args = self.update_args(
            plot_type=PlotType.Scatter,
            new_args=user_plt_args_dct,
            ax_args=ax_args0,
            plt_args=plt_args0,
        )

        # fig = plt.gcf()
        # ax = fig.axes

        if color is not None:
            sp = self._ax.scatter(x, y, c=color, s=plt_args["size"])
            cbar = self._fig.colorbar(sp)
            cbar.set_label(plt_args["cbar_label"])

        else:
            self._ax.scatter(x, y, s=plt_args["size"])

        update_plot(ax_args=ax_args, ax=self._ax)
        # plt.draw()
        # fig.canvas.draw_idle()

    # TODO: Specific to 'single_plot'
    # TODO: This function is doing too much.
    def single_output(self, processor: "Processor") -> None:
        """Save data into a file and/or generate pictures.

        Parameters
        ----------
        processor
        """
        warnings.warn(
            "Use function 'save_to_file' and 'single_plot'.", DeprecationWarning
        )

        assert self._single_plot is not None

        # if not self.save_data_to_file:
        #     self.save_data_to_file = [{"detector.image.array": ["fits"]}]

        self.save_to_file(processor)

        self.single_to_plot(processor)

    # TODO: Specific to 'single_plot'
    def single_to_plot(self, processor: "Processor") -> None:
        """Generate picture(s).

        Parameters
        ----------
        processor
        """
        assert self._single_plot is not None

        self.user_plt_args = None
        x = processor.detector.photon.array  # todo: default plots with plot_args?
        y = processor.detector.image.array
        color = None

        if "plot_args" in self._single_plot:
            plot_args = self._single_plot["plot_args"]  # type: PlotArguments
            self.user_plt_args = plot_args

        if "x" in self._single_plot:
            x = processor.get(self._single_plot["x"])

        if "y" in self._single_plot:
            y = processor.get(self._single_plot["y"])

        if "plot_type" not in self._single_plot:
            raise NotImplementedError

        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)

        x = x.flatten()
        y = y.flatten()

        if self._single_plot["plot_type"] == "graph":
            self.plot_graph(x=x, y=y)  # type: plt.Figure
            fname = "graph_??"

        elif self._single_plot["plot_type"] == "histogram":
            self.plot_histogram(y)
            fname = "histogram_??"

        elif self._single_plot["plot_type"] == "scatter":
            self.plot_scatter(x, y, color)
            fname = "scatter_??"

        else:
            raise KeyError()

        self.save_plot(filename=fname)

        # plt.close()

    def save_to_file(self, processor: "Processor") -> None:
        """Save outputs into file(s).

        Parameters
        ----------
        processor : Processor
        """
        save_methods = {
            "fits": self.save_to_fits,
            "hdf": self.save_to_hdf,
            "npy": self.save_to_npy,
            "txt": self.save_to_txt,
            "csv": self.save_to_csv,
            "png": self.save_to_png,
        }  # type: t.Dict[str, SaveToFile]

        for dct in self.save_data_to_file:  # type: t.Dict[str, t.List[str]]
            # Get first entry of `dict` 'item'
            first_item, *_ = dct.items()

            obj, format_list = first_item

            data = processor.get(obj)  # type: np.ndarray

            if format_list is not None:
                for out_format in format_list:
                    func = save_methods[out_format]  # type: SaveToFile
                    _ = func(data=data, name=obj)

    # TODO: Specific to 'calibration_plot'
    def champions_plot(
        self, results: dict, champions_file: Path, island_id: int
    ) -> None:
        """TBW."""
        data = np.loadtxt(champions_file)
        generations = data[:, 0].astype(int)
        title = "Calibrated parameter: "
        items = list(results.items())
        a = 1
        for item in items:
            plt_args = {"xlabel": "generation", "linestyle": "-", "sci_y": True}
            key = item[0]
            param_value = item[1]
            param_name = key[slice(key.rfind(".") + 1, None)]
            plt_args["ylabel"] = param_name
            if param_name == "fitness":
                plt_args["title"] = "Champion fitness"
                plt_args["color"] = "red"
                plt_args["ylabel"] = "fitness"

            elif param_name == "island":
                continue

            else:
                if key.rfind(".arguments") == -1:
                    mdn = key[: key.rfind("." + param_name)]
                else:
                    mdn = key[: key.rfind(".arguments")]
                model_name = mdn[slice(mdn.rfind(".") + 1, None)]
                plt_args["title"] = title + model_name + " / " + param_name
                plt_args["ylabel"] = param_name

            b = 1
            if isinstance(param_value, Number):
                column = data[:, a]
                self.plot_graph(x=generations, y=column, args=plt_args)

            elif isinstance(param_value, np.ndarray):
                b = len(param_value)
                column = data[:, slice(a, a + b)]
                self.plot_graph(x=generations, y=column, args=plt_args)
                self._fig.legend(["index " + str(i) for i in range(b)])

            self.save_plot(filename=f"calibrated_{param_name!s}_id{island_id!s}")
            a += b

    # TODO: Specific to 'calibration_plot' ??
    def population_plot(
        self, results: dict, population_file: Path, island_id: int
    ) -> None:
        """TBW."""
        assert self.calibration_plot

        data = np.loadtxt(population_file)
        fitnesses = np.log10(data[:, 1])
        a, b = 2, 1  # 1st parameter and fitness
        if self.calibration_plot["population_plot"]:
            if "columns" in self.calibration_plot["population_plot"]:
                col = self.calibration_plot["population_plot"]["columns"]
                a, b = col[0], col[1]
        x = data[:, a]
        y = data[:, b]

        plt_args = {
            "title": "Population of the last generation",
            "size": 8,
            "cbar_label": "log(fitness)",
        }
        if b == 0:
            plt_args["ylabel"] = "generation"
        elif b == 1:
            plt_args["ylabel"] = "fitness"
        else:
            plt_args["ylabel"] = "champions file column #" + str(b)
        if a == 0:
            plt_args["xlabel"] = "generation"
        elif a == 1:
            plt_args["xlabel"] = "fitness"
        else:
            plt_args["xlabel"] = "champions file column #" + str(a)

        if a == 1 or b == 1:
            plt_args["sci_y"] = True
            self.plot_scatter(x, y, args=plt_args)
        else:
            self.plot_scatter(x, y, color=fitnesses, args=plt_args)

        self.save_plot(filename=f"population_id{island_id}")

    # TODO: Specific to 'single_plot'
    def calibration_outputs(self, processor_list: "t.List[Processor]") -> None:
        """TBW."""
        if self.save_data_to_file is not None:
            for processor in processor_list:
                self.save_to_file(processor)

                if self._single_plot:
                    self.single_to_plot(processor)

    # TODO: Specific to 'calibration_plot'
    def calibration_plots(self, results: dict) -> None:
        """TBW."""
        assert self.calibration_plot

        if self.calibration_plot:
            if "champions_plot" in self.calibration_plot:
                self.user_plt_args = None

                if self.calibration_plot["champions_plot"]:
                    if "plot_args" in self.calibration_plot["champions_plot"]:
                        self.user_plt_args = self.calibration_plot["champions_plot"][
                            "plot_args"
                        ]

                for iid, file_ch in enumerate(
                    self.output_dir.glob("champions_id*.out")
                ):
                    self.champions_plot(
                        results=results, champions_file=file_ch, island_id=iid
                    )

            if "population_plot" in self.calibration_plot:
                self.user_plt_args = None
                if self.calibration_plot["population_plot"]:
                    if "plot_args" in self.calibration_plot["population_plot"]:
                        self.user_plt_args = self.calibration_plot["population_plot"][
                            "plot_args"
                        ]

                for iid, file_pop in enumerate(
                    self.output_dir.glob("population_id*.out")
                ):
                    self.population_plot(
                        results=results, population_file=file_pop, island_id=iid
                    )

    # TODO: Specific to 'calibration_plot' ??
    def fitting_plot(
        self, target_data: np.ndarray, simulated_data: np.ndarray, data_i: int
    ) -> None:
        """TBW."""
        assert self.calibration_plot

        if self.calibration_plot:
            if "fitting_plot" in self.calibration_plot:
                self._fig.plot(target_data, ".-", label=f"target data #{data_i}")
                self._fig.plot(simulated_data, ".-", label=f"simulated data #{data_i}")
                self._fig.canvas.draw_idle()

    # TODO: Specific to 'calibration_plot' ??
    def fitting_plot_close(self, result_type: ResultType, island: int) -> None:
        """TBW."""
        assert self.calibration_plot
        assert isinstance(island, int)

        if self.calibration_plot:
            if "fitting_plot" in self.calibration_plot:
                self.user_plt_args = None
                if "plot_args" not in self.calibration_plot["fitting_plot"]:
                    raise RuntimeError

                self.user_plt_args = self.calibration_plot["fitting_plot"]["plot_args"]

                user_plt_args_dct = None  # type: t.Optional[dict]
                if isinstance(self.user_plt_args, PlotArguments):
                    user_plt_args_dct = self.user_plt_args.to_dict()

                args = {
                    "title": (
                        f"Target and Simulated ({result_type}) data, "
                        f"island {island}"
                    )
                }
                ax_args0, plt_args0 = self.update_args(
                    plot_type=PlotType.Graph, new_args=args
                )
                ax_args, plt_args = self.update_args(
                    plot_type=PlotType.Graph,
                    new_args=user_plt_args_dct,
                    ax_args=ax_args0,
                    plt_args=plt_args0,
                )
                update_plot(ax_args=ax_args, ax=self._fig.axes)
                # plt.legend()
                self._fig.legend()

                self.save_plot(filename=f"fitted_datasets_id{island}")

    # TODO: Specific to 'parametric_plot' ?
    def params_func(self, param: "ParametricAnalysis") -> None:
        """Extract all parametric keys from `param`."""
        assert self.parameter_keys is not None

        # TODO: Re-initialized 'self.parameters' ??

        for var in param.enabled_steps:  # type: ParameterValues
            if var.key not in self.parameter_keys:
                self.parameter_keys += [var.key]

        if self.save_parameter_to_file and self.save_parameter_to_file["parameter"]:
            for par in self.save_parameter_to_file["parameter"]:
                if par is not None and par not in self.parameter_keys:
                    self.parameter_keys += [par]

    # TODO: This function should be moved in `ParametricAnalysis`
    # TODO: Specific to 'parametric_plot' ?
    def extract_func(self, processor: "Processor") -> Result:
        """TBW."""
        assert self.parameter_keys is not None

        # self.single_output(processor.detector)    # TODO: extract other things (optional)

        res_row = []  # type: t.List[np.ndarray]
        for key in self.parameter_keys:
            value = processor.get(key)  # type: np.ndarray
            res_row.append(value)

        # Extract all parameters keys from 'proc'
        # all_attr_getters = operator.attrgetter(self.parameter_keys)  # type: t.Callable
        # res_row = all_attr_getters(processor)  # type: t.Tuple[t.Any, ...]

        # TODO: Refactor this
        plt_row = []  # type: t.List[np.ndarray]
        if self.parametric_plot:
            for key in [self.parametric_plot.x, self.parametric_plot.y]:
                if key is not None:
                    value = processor.get(key)
                    plt_row.append(value)

        return Result(
            result=np.array(res_row, dtype=np.float),
            plot=np.array(plt_row, dtype=np.float),
        )

    # TODO: Specific to 'parametric_mode' ?
    def merge_func(self, result_list: t.List[Result]) -> np.ndarray:
        """TBW."""
        assert self.parameter_keys is not None

        if self.save_parameter_to_file:
            result_array = np.array([k.result for k in result_list])
            save_methods = {
                "npy": self.save_to_npy,
                "txt": self.save_to_txt,
                "csv": self.save_to_csv,
            }  # type: t.Dict[str, SaveToFile]

            for out_format in self.save_parameter_to_file["file_format"]:
                func = save_methods[out_format]  # type: SaveToFile
                file = func(data=result_array, name="parameters")  # type: Path

                if file.suffix in (".txt", ".csv"):
                    with file.open("r+") as f:
                        content = f.read()
                        f.seek(0, 0)
                        f.write(
                            "# "
                            + "".join([pp + " // " for pp in self.parameter_keys])
                            + "\n"
                            + content
                        )
        plot_array = np.array([k.plot for k in result_list])  # type: np.ndarray
        return plot_array

    # TODO: Specific to 'parametric_plot' ?
    def plotting_func(self, plot_array: np.ndarray) -> None:
        """TBW.

        Parameters
        ----------
        plot_array : array
        """
        if not self.parametric_plot:
            raise RuntimeError

        x_key = self.parametric_plot.x  # type: str
        y_key = self.parametric_plot.y  # type: str

        self.user_plt_args = self.parametric_plot.plot_args

        x = plot_array[:, 0]  # type: np.ndarray
        y = plot_array[:, 1]  # type: np.ndarray

        par_name = x_key[
            x_key[: x_key[: x_key.rfind(".")].rfind(".")].rfind(".") + 1 :  # noqa: E203
        ]
        res_name = y_key[
            y_key[: y_key[: y_key.rfind(".")].rfind(".")].rfind(".") + 1 :  # noqa: E203
        ]

        args = {"xlabel": par_name, "ylabel": res_name}

        if isinstance(x, np.ndarray):
            x = x.flatten()
        if isinstance(y, np.ndarray):
            y = y.flatten()

        self.plot_graph(x=x, y=y, args=args)
        self.save_plot(filename="parametric_??")

    def update_args(
        self,
        plot_type: PlotType,
        new_args: t.Optional[dict] = None,
        ax_args: t.Optional[dict] = None,
        plt_args: t.Optional[dict] = None,
    ) -> t.Tuple[dict, dict]:
        """TBW."""
        if new_args is None:
            new_args = {}

        if ax_args is None:
            ax_args = self.default_ax_args.to_dict()

        if plt_args is None:
            if plot_type is PlotType.Histogram:
                plt_args = self.default_hist_args.copy()
            elif plot_type is PlotType.Graph:
                plt_args = self.default_plot_args.copy()
            elif plot_type is PlotType.Scatter:
                plt_args = self.default_scatter_args.copy()
            else:
                raise ValueError

        for key in new_args:
            if key in plt_args.keys():
                plt_args[key] = new_args[key]
            elif key in ax_args.keys():
                ax_args[key] = new_args[key]
            else:
                raise KeyError('Not valid plotting key in "plot_args": "%s"' % key)

        return ax_args, plt_args


def show_plots() -> None:
    """Close last empty canvas and show all the previously created figures."""
    plt.close()
    plt.show()


def update_plot(ax_args: dict, ax: plt.Axes) -> None:
    """TBW.

    Parameters
    ----------
    ax_args
    ax
    """
    ax.set_xlabel(ax_args["xlabel"])
    ax.set_ylabel(ax_args["ylabel"])

    ax.set_xscale(ax_args["xscale"])
    ax.set_yscale(ax_args["yscale"])

    ax.set_xlim(ax_args["xlim"][0], ax_args["xlim"][1])
    ax.set_ylim(ax_args["ylim"][0], ax_args["ylim"][1])

    ax.set_title(ax_args["title"])

    if ax_args["axis"]:
        # TODO: Fix this
        raise NotImplementedError
        # plt.axis(ax_args["axis"])

    if ax_args["xticks"]:
        ax.set_xticks(ax_args["xticks"])
    if ax_args["yticks"]:
        ax.set_yticks(ax_args["yticks"])

    # TODO: Enable this
    ax.grid(ax_args["grid"])

    if ax_args["sci_x"]:
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
    if ax_args["sci_y"]:
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))


# TODO: Refactor this function
def update_fits_header(
    header: dict, key: t.Union[str, list, tuple], value: t.Any
) -> None:
    """TBW.

    Parameters
    ----------
    header
    key
    value
    """
    if isinstance(value, (str, int, float)):
        result = value  # type: t.Union[str, int, float]
    else:
        result = repr(value)

    if isinstance(result, str):
        result = result[0:24]

    if isinstance(key, (list, tuple)):
        key = "/".join(key)

    key = key.replace(".", "/")[0:36]
    header[key] = value


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

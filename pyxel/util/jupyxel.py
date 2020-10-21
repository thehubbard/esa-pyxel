#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Tools for jupyter notebook visualization."""

import typing as t

import matplotlib.pyplot as plt
import numpy as np

# # Display methods for detector objects in Jupyter notebook
from IPython.display import Markdown, display
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pyxel.data_structure import Image, Photon, Pixel, Signal
from pyxel.detectors import Detector
from pyxel.pipelines import Processor
from pyxel.pipelines.model_group import ModelGroup

# ----------------------------------------------------------------------------------------------
# Those two methods are used to display the contents of the configuration once loaded in pyxel


def display_config(cfg: dict, only: str = "all") -> None:
    """Display configuration.

    Parameters
    ----------
    cfg: dict
    only: str

    Returns
    -------
    None
    """
    for k in cfg:
        if (only not in cfg.keys()) & (only != "all"):
            error = "Config file only contains following keys: " + str(cfg.keys())
            display(Markdown("<font color=red>" + error + "</font>"))
            break
        elif (only == k) & (only != "all"):
            display(Markdown("## <font color=blue>" + k + "</font>"))
            display(Markdown("\t" + str(cfg[k])))
            if isinstance(cfg[k].__dict__, dict):
                display_dict(cfg[k].__dict__)
        elif only == "all":
            display(Markdown("## <font color=blue>" + k + "</font>"))
            display(Markdown("\t" + str(cfg[k])))
            if isinstance(cfg[k].__dict__, dict):
                display_dict(cfg[k].__dict__)


def display_dict(cfg: dict) -> None:
    """Display configuration dictionary.

    Parameters
    ----------
    cfg: dict

    Returns
    -------
    None
    """
    for k in cfg:
        display(Markdown("#### <font color=#0088FF>" + k + "</font>"))
        display(Markdown("\t" + str(cfg[k])))


# ----------------------------------------------------------------------------------------------
# This method will display the parameters of a specific model


def display_model(
    pipeline_container: t.Union[Processor, dict], model_name: str
) -> None:
    """Display model from configuration dictionary or Processor object.

    Parameters
    ----------
    pipeline_container: Processor or dict
    model_name: str

    Returns
    -------
    None
    """
    model_found = False
    if isinstance(pipeline_container, Processor):
        pipeline = pipeline_container.pipeline
    else:
        pipeline = pipeline_container["pipeline"]
    for value in pipeline.__dict__.values():
        if isinstance(value, ModelGroup):
            # value is a list of ModelFunction namespaces
            models_list = value.__dict__["models"]
            for model_namespace in models_list:
                # print(np.where([name for name in modelNamespace.name] == model_name))
                if model_name in model_namespace.name:
                    display(Markdown("## <font color=blue>" + model_name + "</font>"))
                    display(
                        Markdown(
                            "".join(
                                [
                                    "Model ",
                                    model_name,
                                    " enabled? ",
                                    str(model_namespace.enabled),
                                ]
                            )
                        )
                    )
                    display_dict(model_namespace.arguments)
                    model_found = True

    if not model_found:
        display(Markdown("<font color=red>" + model_name + " not found</font>"))


def change_modelparam(
    processor: Processor, model_name: str, argument: str, changed_value: t.Any
) -> None:
    """Change model parameter.

    Parameters
    ----------
    processor: Processor
    model_name: str
    argument:str
    changed_value

    Returns
    -------
    None
    """
    display(Markdown("## <font color=blue>" + model_name + "</font>"))
    for value in processor.__dict__.values():
        if isinstance(value, ModelGroup):
            for model in value.__dict__["models"]:
                if model.name == model_name:
                    try:
                        model.arguments[argument] = changed_value
                    except KeyError:
                        print(model_name, "possess no argument named: ", value)


def set_modelstate(processor: Processor, model_name: str, state: bool = True) -> None:
    """Change model state (true/false).

    Parameters
    ----------
    processor: Processor
    model_name: str
    state: bool

    Returns
    -------
    None
    """
    display(Markdown("## <font color=blue>" + model_name + "</font>"))
    for value in processor.pipeline.__dict__.values():
        if isinstance(value, ModelGroup):
            for model in value.__dict__["models"]:
                if model.name == model_name:
                    try:
                        model.enabled = state
                        display(
                            Markdown(
                                "".join(
                                    [
                                        "Model ",
                                        model_name,
                                        " enabled? ",
                                        str(model.enabled),
                                    ]
                                )
                            )
                        )
                    except KeyError:
                        print(model_name, "possess no argument named: ", value)


# ----------------------------------------------------------------------------------------------
# These method are used to display the detector object (all of the array Phton, pixel, signal and image)


def display_array(data: np.ndarray, axes: t.List[plt.axes], **kwargs: str) -> None:
    """For a pair of axes, display the image on the first one, the histogram on the second.

    Parameters
    ----------
    data: ndarray
        A 2D np.array.
    axes: list
        A list of two axes in a figure.

    Returns
    -------
    None
    """
    im = axes[0].imshow(data)
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    divider = make_axes_locatable(axes[0])
    cax1 = divider.append_axes("left", size="5%", pad=0.05)
    axes[0].set_title(kwargs["label"])
    plt.colorbar(im, cax=cax1)
    cax1.yaxis.set_label_position("left")
    cax1.yaxis.set_ticks_position("left")
    axes[1].hist(data.flatten(), bins=50, **kwargs)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.5)


def display_detector(
    detector: Detector, array: t.Union[None, Photon, Pixel, Signal, Image] = None
) -> None:
    """Display detector.

    Parameters
    ----------
    detector: Detector
    array: str

    Returns
    -------
    None
    """
    if array is not None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        display_array(array.array, axes, label=str(array).split("<")[0])
    else:
        arrays = [detector.photon, detector.pixel, detector.signal, detector.image]

        fig, axes = plt.subplots(len(arrays), 2, figsize=(15, 6 * len(arrays)))

        for idx, data in enumerate(arrays):
            display_array(data.array, axes[idx], label=str(data).split("<")[0])

    plt.show()


# ----------------------------------------------------------------------------------------------
# These method are used to display the detector memory


def display_persist(persist_dict: dict) -> None:
    """Plot all trapped charges using the memory dict.

    Parameters
    ----------
    persist_dict: dict

    Returns
    -------
    None
    """
    trapped_charges = [trapmap for trapmap in persist_dict.values()]

    fig, axes = plt.subplots(
        len(trapped_charges), 2, figsize=(10, 5 * len(trapped_charges))
    )
    for ax, trapmap, keyw in zip(
        axes, trapped_charges, [i.replace("_", "\n") for i in persist_dict.keys()]
    ):
        display_array(trapmap, ax, label=keyw)

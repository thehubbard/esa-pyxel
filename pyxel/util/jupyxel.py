#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt

# # Display methods for detector objects in Jupyter notebook
from IPython.display import Markdown, display
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pyxel.pipelines.model_group import ModelGroup
import typing as t
from pyxel.pipelines import Processor

# ----------------------------------------------------------------------------------------------
# Those two methods are used to display the contents of the configuration once loaded in pyxel


def display_config(cfg, only="all"):
    """DocString"""
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


def display_dict(cfg):
    for k in cfg:
        display(Markdown("#### <font color=#0088FF>" + k + "</font>"))
        display(Markdown("\t" + str(cfg[k])))


# ----------------------------------------------------------------------------------------------
# This method will display the parameters of a specific model


def display_model(cfg, model_name: str) -> None:
    model_found = False
    for value in cfg["pipeline"].__dict__.values():
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


# TODO: change cfg to processor
# TODO: change type to isinstance
# TODO: fix flake8
def change_modelparam(cfg: dict, model_name: str, argument: str, changed_value: t.Any) -> None:
    display(Markdown("## <font color=blue>" + model_name + "</font>"))
    for value in cfg["pipeline"].__dict__.values():
        if isinstance(value, ModelGroup):
            for model in value.__dict__["models"]:
                if model.name == model_name:
                    try:
                        model.arguments[argument] = changed_value
                    except KeyError:
                        print(model_name, "possess no argument named: ", value)


def set_modelstate(cfg: dict, model_name: str, state: bool = True) -> None:
    display(Markdown("## <font color=blue>" + model_name + "</font>"))
    for value in cfg["pipeline"].__dict__.values():
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


def display_array(data, axes, **kwargs):
    """For a pair of axes, will display the image on the first one, the histogram on the second
    :input: data, a 2D np.array
            axes, a list of two axes in a figure
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


def display_detector(detector, array="all"):
    """DocString display_detector"""
    if array != "all":
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        display_array(array.array, axes, label=str(array).split("<")[0])
    else:
        arrays = [detector.photon, detector.pixel, detector.signal, detector.image]

        fig, axes = plt.subplots(len(arrays), 2, figsize=(15, 6 * len(arrays)))
        [
            display_array(data.array, axes[idx], label=str(data).split("<")[0])
            for idx, data in enumerate(arrays)
        ]

    plt.show()


# ----------------------------------------------------------------------------------------------
# These method are used to display the detector memory


def display_persist(persist_dict):
    """Uses the meomry dict to plot all trapped charges"""
    trapped_charges = [trapmap for trapmap in persist_dict.values()]

    fig, axes = plt.subplots(
        len(trapped_charges), 2, figsize=(10, 5 * len(trapped_charges))
    )
    for ax, trapmap, keyw in zip(
        axes, trapped_charges, [i.replace("_", "\n") for i in persist_dict.keys()]
    ):
        display_array(trapmap, ax, label=keyw)

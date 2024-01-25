#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
"""Notebook functions."""

# flake8: noqa
from .jupyxel import (
    display_config,
    display_dict,
    display_model,
    change_modelparam,
    display_array,
    display_detector,
    display_persist,
    display_scene,
)
from .html_representation import display_html

from .calibration import (
    display_calibration_inputs,
    display_simulated,
    display_evolution,
    champion_heatmap,
    optimal_parameters,
)

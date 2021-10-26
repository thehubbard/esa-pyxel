.. _running_modes:

=============
Running modes
=============

There are three running modes in Pyxel:

- **Exposure mode**: simulation of a single exposure, at a single or with incrementing readout times (quick look/ health check, simulation of non-destructive readout mode and time-dependent effects),
- **Observation mode**: multiple exposure pipelines looping over a range of model or detector parameters (sensitivity analysis),
- **Calibration mode**: optimize model or detector parameters to fit target data sets using a user-defined fitness function/figure of merit (model fitting, instrument optimization),

Click below for more information on the modes and corresponding YAML configurations.

.. toctree::
    running_modes/exposure_mode.rst
    running_modes/observation_mode.rst
    running_modes/calibration_mode.rst

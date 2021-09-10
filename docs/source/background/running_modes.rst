.. _running_modes:

=============
Running modes
=============

There are four running modes in Pyxel:

- **Single mode**: one image in, one image out, single pipeline run (quick look/ health check),
- **Parametric mode**: pipeline is run multiple times looping over a range of model or detector parameters (sensitivity analysis),
- **Calibration mode**: optimize model or detector parameters to fit target data sets using a user-defined fitness function/figure of merit (model fitting, instrument optimization),
- **Dynamic mode**: the pipeline is run N times incrementing time, saving detector attributes (simulation of non-destructive readout mode, and time-dependent effects)

Click below for more information on the modes and corresponding YAML configurations.

.. toctree::
    running_modes/single_mode.rst
    running_modes/parametric_mode.rst
    running_modes/dynamic_mode.rst
    running_modes/calibration_mode.rst

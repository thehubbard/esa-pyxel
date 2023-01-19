---
title: What are the different running modes in Pyxel?
---

There are three [running modes](https://esa.gitlab.io/pyxel/doc/stable/background/running_modes.html) in Pyxel:

[Exposure mode:](https://esa.gitlab.io/pyxel/doc/stable/background/running_modes/exposure_mode.html#exposure-mode) is used for a simulation of a single exposure, at a single or with incrementing readout times 
(quick look/ health check, simulation of non-destructive readout mode and time-dependent effects).
[Observation mode:](https://esa.gitlab.io/pyxel/doc/stable/background/running_modes/observation_mode.html) consists of multiple exposure pipelines looping over a range of model or detector parameters (sensitivity analysis).
[Calibration mode:](https://esa.gitlab.io/pyxel/doc/stable/background/running_modes/calibration_mode.html) is used to optimize model or detector parameters to fit target data sets using a user-defined fitness function/figure of merit 
(model fitting, instrument optimization).
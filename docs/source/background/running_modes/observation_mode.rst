.. _parametric_mode:

================
Observation mode
================

The observation mode of Pyxel can automatically change the value of any
detector or model parameter to simulate an observation or make a sensitivity analysis for any parameter.
Iterating over different start times of exposures is also possible.

The variable parameter have to be defined in the YAML
configuration file with ranges or lists. The framework generates and runs
a stack of different Detector objects and pipelines.

At the end, the user can plot and analyze the data
in function of the variable parameter.

There are three different modes of defining parameters:

Sequential
----------

.. code-block:: yaml

  # YAML config file for Observation mode (sequential)

  observation:

    mode: sequential
    parameters:
      - key:      pipeline.charge_generation.tars.arguments.particle_number
        values:   [1, 2, 3]
        enabled:  true
      - key:      pipeline.photon_generation.illumination.arguments.level
        values:   range(0, 300, 100)
        enabled:  true

| The yaml file will start 6 runs in this order:
| number=1, number=2, number=3, level=0, level=100, level=200

The default values for 'number' and 'level' are defined as the arguments
of the specific models in the pipeline part of the yaml config file.

Product
-------

.. code-block:: yaml

  # YAML config file for Observation mode (product)

  observation:

    mode: product
    parameters:
      - key:      pipeline.charge_generation.tars.arguments.particle_number
        values:   [1, 2, 3]
        enabled:  true
      - key:      pipeline.photon_generation.illumination.arguments.level
        values:   range(0, 300, 100)
        enabled:  true

| The yaml file will start 9 runs in this order:
| (number=1, level=0), (number=1, level=100), (number=1, level=200),
| (number=2, level=0), (number=2, level=100), (number=2, level=200),
| (number=3, level=0), (number=3, level=100), (number=3, level=200)

The default values for 'number' and 'level' are defined as the arguments
of the specific models in the pipeline part of the yaml config file.

Custom
------

.. code-block:: yaml

  # YAML config file for Observation mode (custom)

  observation:

    mode:  custom
    from_file:        'outputs/calibration_champions.out'
    column_range:     [2, 17]
    parameters:
      - key:      detector.characteristics.amp
        values:   _
      - key:      pipeline.charge_transfer.cdm.arguments.tr_p
        values:   [_, _, _, _]
      - key:      pipeline.charge_transfer.cdm.arguments.nt_p
        values:   [_, _, _, _]
      - key:      pipeline.charge_transfer.cdm.arguments.sigma_p
        values:   [_, _, _, _]
      - key:      pipeline.charge_transfer.cdm.arguments.beta_p
        values:   _
      - key:      detector.environment.temperature
        values:   _

The parameter values (int, float or str) indicated with with '_' character,
and all are read and changed in parallel from an ASCII file defined
with ``from_file``.

Can be used for example to read results of calibration running mode
containing the champion parameter set for each generation, and create one
output fits image for each generation to see the evolution.
.. _running_modes:

=============
Running modes
=============

.. _single_mode:

Single mode
===========

Running Pyxel in Single mode can be used to get a single image with
the detector effects defined in either the configuration file
or the GUI.

.. code-block:: yaml

  # YAML config file for Single mode

    single


..
    either with or without a time dependent readout. In the former case,
    time evolution of images is available as well.


.. _parametric_mode:

Parametric mode
===============

The parametric mode of Pyxel can automatically change the value of any
detector or model parameter to make a sensitivity analysis for any parameter.

The variable parameter have to be defined in the YAML
configuration file with ranges or lists. The framework generates and runs
a stack of different Detector objects and pipelines.

At the end, the user can plot and analyze the data
in function of the variable parameter.

Sequential
----------

.. code-block:: yaml

  # YAML config file for Parametric mode (sequential)

  parametric:

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

  # YAML config file for Parametric mode (product)

  parametric:

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

  # YAML config file for Parametric mode (custom)

  parametric:

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

The parametric values (int, float or str) indicated with with '_' character,
and all are read and changed in parallel from an ASCII file defined
with ``from_file``.

Can be used for example to read results of calibration running mode
containing the champion parameter set for each generation, and create one
output fits image for each generation to see the evolution.

.. _calibration_mode:

Calibration mode
================

The purpose of the Calibration mode is to find the optimal input arguments
of models or optimal detector attributes based on a
target dataset the models or detector behaviour shall reproduce.

..
    The architecture contains a data
    comparator function to compare simulated and measured data, then via a
    feedback loop, a function readjusts the model parameters (this function
    can be user defined).
    The Detection pipelines are re-run with the modified
    Detector objects. This iteration continues until reaching the convergence,
    i.e. we get a calibrated model fitted to the real, measured dataset.


.. code-block:: yaml

  # YAML config file for Calibration mode


  calibration:

    mode: pipeline                                # single_model

    result_type:      image                       # pixel # signal # image
    result_fit_range: [0, 20, 0, 30]

    target_data_path: [data/target.fits']         #  <*.npy> <*.fits> <ascii>
    target_fit_range: [10, 30, 20, 50]

    weighting_path:   ['data/weights.fits']

    fitness_function:
      func: pyxel.calibration.fitness.sum_of_abs_residuals
      arguments:

    algorithm:
      type:            sade                       # sga # nlopt
      generations:     20
      population_size: 100
      variant:         2

    seed:              1321

    parameters:
      - key:  detector.characteristics.amp
        values: _
        logarithmic: false
        boundaries: [1., 10.]
      - key:  pipeline.charge_transfer.cdm.arguments.tr_p
        values: [_, _, _, _]
        logarithmic: true
        boundaries: [1.e-3, 2.]
      - key:  pipeline.charge_transfer.cdm.arguments.nt_p
        values: [_, _, _, _]
        logarithmic: true
        boundaries: [1.e-2, 1.e+1]
      - key:  pipeline.charge_transfer.cdm.arguments.beta_p
        values: _
        logarithmic: false
        boundaries: [0., 1.]


.. _dynamic_mode:

Dynamic mode
============

The purpose of the Dynamic mode is to execute the same pipeline
on the same detector object several times. It uses the parameter `steps`
to count the number of iteration of the detector through the pipeline.
The parameter `t_step` is the time in between steps, that can be used
in the models that use time-dependant computation.

The `non-destructive` mode is used to avoid reseting the detector object
at each iteration of the detector through the pipeline.

.. code-block:: yaml

  # YAML config file for Dynamic mode

  dynamic:
    non_destructive_readout:  true
    steps: 10
    t_step: 0.5

    outputs:
      output_folder: 'output'

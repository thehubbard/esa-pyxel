.. _calibration_mode:

================
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
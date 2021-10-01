.. _dynamic_mode:

============
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
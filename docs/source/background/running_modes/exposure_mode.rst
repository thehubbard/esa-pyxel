.. _exposure_mode:

=============
Exposure mode
=============

Single readout time
-------------------

Running Pyxel in Exposure mode can be used to get a single image with
the detector effects defined in either the configuration file.
If multiple readout times are not specified, the default readout time is 1 second.

.. code-block:: yaml

    exposure:
      outputs:
        output_folder: 'output'

..

Multiple readout times (time-domain simulation)
-----------------------------------------------

The purpose of having multiple readout times is to execute the same pipeline
on the same detector object several times. In that case, time evolution of images is available as well.
Readout times have to be specified in the ``YAML`` file like shown below.
Users can also set the readout to non-destructive or destructive, set start time or upload times from a file.

The `non-destructive` mode is used to avoid resetting the detector object
and emptying the data at each iteration of the detector through the pipeline.

.. code-block:: yaml

  exposure:

    readout:
      times: numpy.linspace(1, 20, 50)
      non_destructive:  true

    outputs:
      output_folder: 'output'
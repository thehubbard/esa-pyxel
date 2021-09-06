=============
Running Pyxel
=============

Pyxel can be run either from command line or used as a library, for example in Jupyter notebooks.

Running Pyxel from command line
===============================

To run Pyxel on your local computer, simply run it from the command-line:

.. code-block:: bash

    $ python pyxel/run.py -c input.yaml

    or

    $ pyxel --config input.yaml

    or

    $ python -m pyxel --config input.yaml


where

======  ===============  =======================================  ========
``-c``  ``--config``     defines the path of the input YAML file  required
``-s``  ``--seed``       defines a seed for random number         optional
                         generator
``-v``  ``--verbosity``  increases the output verbosity (-v/-vv)  optional
``-V``  ``--version``    prints the version of Pyxel              optional
======  ===============  =======================================  ========

Running Pyxel in jupyter notebooks
==================================

.. code-block:: python


    from pyxel.configuration import load
    from pyxel.run import single_mode

    configuration = load("configuration.yaml")
    single = configuration.single
    detector = configuration.ccd_detector
    pipeline = configuration.pipeline

    single_mode(single, detector, pipeline)
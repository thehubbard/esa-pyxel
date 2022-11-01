.. _examples:

======================
Tutorials and examples
======================

We recommend you to start with the tutorial available in the form of Jupyter notebooks.
It covers all the basics, the four running modes and adding a new model. Apart from the tutorial,
more examples on running modes and different models are also available. See below for a full list.

All tutorials and examples can be found in a separate public repository
`Pyxel Data <https://gitlab.com/esa/pyxel-data>`_, to access it click on the link below.

.. link-button:: https://gitlab.com/esa/pyxel-data
    :type: url
    :text: To tutorials and examples repository
    :classes: btn-outline-primary btn-block

Once you’ve installed Pyxel, the example repository can be either downloaded directly by clicking on button download
or using Pyxel by running the command:

.. code-block:: console

    pyxel download-examples

Now you can launch JupyterLab to explore them:

.. code-block:: console

    cd pyxel-examples

    jupyter lab

You can run also tutorials and examples without prior installation of Pyxel in a live session here: |Binder|

.. |Binder| image:: https://static.mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gl/esa%2Fpyxel-data/HEAD?urlpath=lab

Tutorial chapters
-----------------

0. `Introduction <https://gitlab.com/esa/pyxel-data/-/blob/master/tutorial/00_introduction.ipynb>`_
1. `First simulation <https://gitlab.com/esa/pyxel-data/-/blob/master/tutorial/01_first_simulation.ipynb>`_
2. `Pyxel configuration and classes <https://gitlab.com/esa/pyxel-data/-/blob/master/tutorial/02_pyxel_configuration_and_classes.ipynb>`_
3. `Create a model <https://gitlab.com/esa/pyxel-data/-/blob/master/tutorial/03_create-model.ipynb>`_
4. `Observation mode <https://gitlab.com/esa/pyxel-data/-/blob/master/tutorial/04_observation_mode.ipynb>`_
5. `Calibration mode <https://gitlab.com/esa/pyxel-data/-/blob/master/tutorial/05_calibration_mode.ipynb>`_
6. `Calibration visualisation <https://gitlab.com/esa/pyxel-data/-/blob/master/tutorial/06_calibration_visualization.ipynb>`_
7. `Simulating multiple readouts <https://gitlab.com/esa/pyxel-data/-/blob/master/tutorial/07_exposure_with_multiple_readouts.ipynb>`_

List of Examples
----------------

**Exposure Mode:**

- `Basic example <https://gitlab.com/esa/pyxel-data/-/blob/master/examples/exposure/exposure.ipynb>`_
- `Persistence in H2RG (time-domain simulation) <https://gitlab.com/esa/pyxel-data/-/blob/master/examples/exposure/exposure_persistence-H4RG.ipynb>`_

**Observation mode:**

- Basic example (`product <https://gitlab.com/esa/pyxel-data/-/blob/master/examples/observation/product.ipynb>`_)
- Basic example (`sequential <https://gitlab.com/esa/pyxel-data/-/blob/master/examples/observation/sequential.ipynb>`_)
- Basic example (`custom <https://gitlab.com/esa/pyxel-data/-/blob/master/examples/observation/custom.ipynb>`_)

**Calibration mode:**

- `Basic calibration example <https://gitlab.com/esa/pyxel-data/-/blob/master/examples/calibration/calibration.ipynb>`_

**Models:**

- `Amplifier crosstalk <https://gitlab.com/esa/pyxel-data/-/blob/master/examples/models/amplifier%20crosstalk/crosstalk.ipynb>`_
- `Inter-pixel capacitance <https://gitlab.com/esa/pyxel-data/-/blob/master/examples/models/inter-pixel%20capacitance/ipc.ipynb>`_

Generic detector pipelines
--------------------------

The Pyxel model library contains models for various types of detectors.
Not all models can be used with all of the detector types
and some specific models are only to be used with a single type of detector.
For this reason and to help new users and non-experts,
generic configuration file templates for different detectors have been included in the Pyxel Data example repository,
together with corresponding Jupyter notebooks.
They include detector properties and pipelines with detector-appropriate sets of models,
pre-filled with realistic model argument values.
They provide a good starting point for simulations of specific detectors and later customization
or iteration with detector engineers and experts.
The generic pipelines are now available for the following types
of detectors: generic :term:`CCD`, generic :term:`CMOS`, Teledyne HxRG,
and Avalanche Photo Diode (:term:`APD`) array detector based on Leonardo’s Saphira detector.

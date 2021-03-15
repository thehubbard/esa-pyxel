.. _examples:

========
Examples
========

All Pyxel examples can be found in a separate public repository
`Pyxel Data <https://gitlab.com/esa/pyxel-data>`_,
containing Jupyter notebooks showcasing different running modes or different models.
You can find the list of available examples below.

Once you’ve installed Pyxel, you can either download them directly
or get a copy of all the examples in the repository by running the commands:

.. code-block:: console

    pyxel --download-examples

Now you can launch JupyterLab to explore them:

.. code-block:: console

    cd pyxel-examples

    jupyter lab

You can run also examples without prior installation of Pyxel in a live session here: |Binder|

.. |Binder| image:: https://static.mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gl/esa%2Fpyxel-data/HEAD?urlpath=lab


List of Examples
----------------

**Single Mode:**

- Basic example
- Amplifier crosstalk model
- Inter-pixel capacitance model

**Parametric mode:**

- Photon transfer curve

**Dynamic mode:**

- Persistence model

**Calibration mode:**

- Charge distortion model
- Charge distortion model with FLEX data

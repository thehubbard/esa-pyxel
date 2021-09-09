.. _introduction:

============
Introduction
============

Pyxel :cite:`2020:prodhomme` is a novel, open-source, modular
Python software framework designed
to host and pipeline models (analytical, numerical, statistical) simulating
different types of detector effects on images produced by Charge-Coupled
Devices (CCD), Monolithic, and Hybrid CMOS imaging sensors.

.. image:: ../_static/pyxel-logo.png
    :alt: logo
    :scale: 20 %
    :align: center

Users can provide one or more input images to Pyxel, set the detector and
model parameters via a user interface (configuration file)
and select which effects to simulate: cosmic rays, detector
Point Spread Function (PSF), electronic noises, Charge Transfer Inefficiency
(CTI), persistence, dark current, charge diffusion, optical effects, etc.
The output is one or more images including the simulated detector effects
combined.

.. figure:: _static/Pyxel-example-transparent.png
    :alt: example
    :align: center

    Examples of output images created using Pyxel.
    Left: original image;
    centre: tracks of cosmic ray protons have been added;
    right: in addition to the cosmic ray protons tracks the effects
    of lower full well capacity and charge transfer inefficiency have been added.


On top of its model hosting capabilities, the framework also provides a set
of basic image analysis tools and an input image generator as well. It also
features a parametric mode to perform parametric and sensitivity analysis,
and a model calibration mode to find optimal values of its parameters
based on a target dataset the model should reproduce.

A majority of Pyxel users are expected to be detector scientists and
engineers working with instruments - using detectors - built for astronomy
and Earth observation, who need to perform detector simulations, for example
to understand laboratory data, to derive detector design specifications for
a particular application, or to predict instrument and mission performance
based on existing detector measurements.

One of the main purposes of this new tool is to share existing resources
and avoid duplication of work. For instance, detector models
developed for a certain project could be reused by
other projects as well, making knowledge transfer easier.

.. note::

    **This publication should be referenced in context of using Pyxel:**

    Thibaut Prod'homme, Frédéric Lemmel, Matej Arko, Benoit Serra, Elizabeth George, Enrico Biancalani, Hans Smit, and David Lucsanyi.
    Pyxel: the collaborative detection simulation framework.
    In Andrew D. Holland and James Beletic, editors, X-Ray, Optical, and Infrared Detectors for Astronomy IX, volume 11454, 26-35.
    International Society for Optics and Photonics, SPIE, 2020. URL: https://doi.org/10.1117/12.2561731


**Tutorials**. Learn about the Pyxel's concepts.
Are you new to Pyxel ? This is the place to start !

* :doc:`install`
* :doc:`running`
* :doc:`examples`

**Before you do anything else, start here** at our :doc:`install` tutorial.

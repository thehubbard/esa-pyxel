.. _architecture:

============
Architecture
============

There are three main elements behind Pyxel's architecture,
the :ref:`running_modes`, the :ref:`detectors` and the :ref:`pipeline`,
each of them represented by a class in the code and corresponding to a section in the configuration file.
See :ref:`apireference` for further information on the three types of classes.
All the three elements are defined before running Pyxel through the input YAML configuration file.
More on the configuration files in :ref:`yaml`

As illustrated below, the detector holds information about the detector properties such as geometry, characteristics,
material and environment. It is also a bucket for storing simulated data,
for example the incoming photons, stored charge in the pixels etc.
This data can be used and edited by any of models in the pipeline.
The detector is the one single object all the models have in common,
so we also store

The pipeline is the the core algorithm, hosting and running the models,
which are grouped into different model groups, levels imitating the working principles of the detector/instrument.

.. figure:: _static/architecture.png
    :scale: 70%
    :alt: architecture
    :align: center

    Image caption.
.. _architecture:

============
Architecture
============

There are three main elements behind Pyxel's architecture,
the :ref:`running_modes`, :ref:`detectors` and the :ref:`pipeline`,
each of them represented by classes in the code.
See :ref:`apireference` for further information on the three types of classes.

All the three elements are defined at the beginning through the input YAML configuration file.


As illustrated below, the detector holds information about the detector properties such as geometry, characteristics,
material and environment. Apart from that, it is also a bucket for storing simulated data,
for example the incoming photons, stored charge in the pixels etc.
This data can be used and edited by models in the pipeline.

The pipeline is the the core algorithm, hosting and running the models,
which are grouped into different model groups, levels imitating the working principles of the detector/instrument.

.. figure:: _static/architecture.png
    :scale: 70%
    :alt: architecture
    :align: center

    Image caption.
============
Architecture
============

There are two main structures behind Pyxel's architecture, the **Detector** and the **Pipeline**,
each one represented by a class. See :ref:`apireference` for further information on the two classes.
As illustrated below, the detector holds information about the detector properties such as geometry, characteristics,
material and environment. Apart from that, it is also a bucket for storing simulated data,
for example the incoming photons, stored charge in the pixels etc. This data can be used by models in the pipeline.

The pipeline is the the core algorithm, hosting and running the models,
which are grouped into different model groups, levels imitating the working principles of the detector/instrument.


.. figure:: _static/architecture.png
    :scale: 70%
    :alt: architecture
    :align: center

.. toctree::

   detectors.rst
   pipelines.rst
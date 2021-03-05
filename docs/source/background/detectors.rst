.. _detectors:

#########
Detectors
#########

According to the YAML file, one CCD or CMOS Detector object is instantiated
for each thread, inheriting from a general (abstract) Detector class.

The created Detector object is the input of the Detection pipeline, which is
passed through all the including models represented by functions. We can
consider the Detector object as a bucket containing all information and data
related to the physical properties of the simulated detector (geometry,
material, environment, characteristics), incident photons, created
charge-carriers and the generated signals we are interested in at the
end of the simulation.


Detector classes and their attributes.

.. _geometry:

Geometry
========

CCD
---

.. autoclass:: pyxel.detectors.CCDGeometry
    :members:
    :inherited-members:
    :undoc-members:
    :show-inheritance:
    :exclude-members:

CMOS
----

.. autoclass:: pyxel.detectors.CMOSGeometry
    :members:
    :inherited-members:
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. _characteristics:

Characteristics
===============

CCD
---

.. autoclass:: pyxel.detectors.CCDCharacteristics
    :members:
    :inherited-members:
    :undoc-members:
    :show-inheritance:
    :exclude-members:

CMOS
----

.. autoclass:: pyxel.detectors.CMOSCharacteristics
    :members:
    :inherited-members:
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. _material:

Material
========

.. autoclass:: pyxel.detectors.Material
    :members:
    :undoc-members:
    :exclude-members:


.. _environment:

Environment
===========

.. autoclass:: pyxel.detectors.Environment
    :members:
    :undoc-members:
    :exclude-members:


.. _optics:

Optics
======

.. autoclass:: pyxel.detectors.Optics
    :members:
    :undoc-members:
    :exclude-members:

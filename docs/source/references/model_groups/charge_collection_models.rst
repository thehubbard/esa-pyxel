.. _charge_collection:

Charge Collection models
========================

.. currentmodule:: pyxel.models.charge_collection

Charge collection models are used to add to and manipulate data in :py:class:`~pyxel.data_structure.Pixel` array
inside the :py:class:`~pyxel.detectors.Detector` object.
The data represents amount of charge stored in each of the pixels.
A charge collection model is necessary to first convert from charge data stored in
:py:class:`~pyxel.data_structure.Charge` class. Multiple models are available to add detector effects after.

Simple collection
-----------------

Simple collection model is the simplest model of charge collection and
necessary to fill up :py:class:`~pyxel.data_structure.Pixel` array when no other collection model is used.
If charge inside :py:class:`~pyxel.data_structure.Charge` class is stored in an ``numpy`` array,
arrays will be the same. If charge is in the form of ``Pandas`` dataframe and
representing 3D point cloud of charges inside the detector,
calling ``array`` property of :py:class:`~pyxel.data_structure.Charge`
will assign charges to the closest pixel and sum the values.

Example of YAML configuration model:

.. code-block:: yaml

    - name: simple_collection
      func: pyxel.models.charge_collection.simple_collection
      enabled: true

.. autofunction:: simple_collection

Simple full well
----------------

.. autofunction:: simple_full_well

Fix pattern noise
-----------------

.. autofunction:: fix_pattern_noise
   :noindex:

Inter-pixel capacitance
-----------------------

.. autofunction:: simple_ipc

Persistence
-----------

.. autofunction:: simple_persistence
.. autofunction:: current_persistence

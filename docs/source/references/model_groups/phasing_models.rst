.. _phasing:

==============
Phasing models
==============

.. currentmodule:: pyxel.models.phasing

.. _phasing_create_store_detector:

Create and Store a detector
===========================

The models :ref:`phasing_save_detector` and :ref:`phasing_load_detector`
can be used respectively to create and to store a :py:class:`~pyxel.detectors.Detector` to/from a file.

These models can be used when you want to store or to inject a :py:class:`~pyxel.detectors.Detector`
into the current :ref:`pipeline`.

.. _phasing_save_detector:

Save detector
-------------

This model saves the current :py:class:`~pyxel.detectors.Detector` into a file.
Accepted file formats are ``.h5``, ``.hdf5``, ``.hdf`` and ``.asdf``.

.. code-block:: yaml

    - name: save_detector
      func: pyxel.models.save_detector
      enabled: true
      arguments:
        filename: my_detector.h5

.. autofunction:: pyxel.models.save_detector
   :noindex:


.. _phasing_load_detector:

Load detector
-------------

This model loads a :py:class:`~pyxel.detectors.Detector` from a file and injects it in the current pipeline.
Accepted file formats are ``.h5``, ``.hdf5``, ``.hdf`` and ``.asdf``.

.. code-block:: yaml

    - name: load_detector
      func: pyxel.models.load_detector
      enabled: true
      arguments:
        filename: my_detector.h5

.. autofunction:: pyxel.models.load_detector
   :noindex:



.. _Pulse processing:

Pulse processing
================

:guilabel:`Charge` → :guilabel:`Phase`

TBW: description, reference, units etc.

Example of YAML configuration model:

.. code-block:: yaml

    - name: pulse_processing
      func: pyxel.models.phasing.pulse_processing
      enabled: true
      arguments:
        wavelength:
        responsivity:
        scaling_factor: 2.5e2

.. note:: This model is specific for the :term:`MKID` detector.

.. autofunction:: pulse_processing
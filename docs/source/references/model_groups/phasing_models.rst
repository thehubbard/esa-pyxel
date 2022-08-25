.. _phasing:

==============
Phasing models
==============

.. currentmodule:: pyxel.models.phasing

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
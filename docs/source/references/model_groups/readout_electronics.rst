.. _readout_electronics:

==========================
Readout Electronics models
==========================

.. currentmodule:: pyxel.models.readout_electronics
.. automodule:: pyxel.models.readout_electronics


Simple digitization
===================

.. autofunction:: simple_digitization

Simple amplification
====================

.. autofunction:: simple_amplifier

DC crosstalk
============

.. autofunction:: dc_crosstalk

AC crosstalk
============

.. autofunction:: ac_crosstalk

SAR ADC
=======

Digitize signal array using SAR (Successive Approximation Register) ADC logic.

Example of the configuration file:

.. code-block:: yaml

    - name: sar_adc
      func: pyxel.models.readout_electronics.sar_adc
      enabled: true
      arguments:
        adc_bits: 16
        range_volt: [0.0, 5.0]

.. autofunction:: sar_adc

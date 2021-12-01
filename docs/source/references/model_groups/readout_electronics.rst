.. _readout_electronics:

==========================
Readout Electronics models
==========================

.. currentmodule:: pyxel.models.readout_electronics

Readout electronics models are used to add TBW.


Simple digitization
===================

Digitize signal array mimicking readout electronics.

Example of the configuration file:

.. code-block:: yaml

    - name: simple_digitization
      func: pyxel.models.readout_electronics.simple_digitization
      enabled: true
      arguments:
        data_type: uint16   # This is optional

.. autofunction:: simple_digitization

Simple amplification
====================

Amplify signal using gain from the output amplifier (in V/V) and
the signal processor (in V/V).


Example of the configuration file:

.. code-block:: yaml

    - name: simple_amplifier
      func: pyxel.models.readout_electronics.simple_amplifier
      enabled: true

.. autofunction:: simple_amplifier

DC crosstalk
============

Apply DC crosstalk signal to detector signal.

Example of the configuration file:

.. code-block:: yaml

    - name: dc_crosstalk
      func: pyxel.models.readout_electronics.dc_crosstalk
      enabled: true
      arguments:
        coupling_matrix: [[1, 0.5, 0, 0], [0.5, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0.5, 1]]
        channel_matrix: [1,2,3,4]
        readout_directions: [1,2,1,2]

.. autofunction:: dc_crosstalk

AC crosstalk
============

Apply AC crosstalk signal to detector signal.

Example of the configuration file:

.. code-block:: yaml

    - name: ac_crosstalk
      func: pyxel.models.readout_electronics.ac_crosstalk
      enabled: true
      arguments:
        coupling_matrix: [[1, 0.5, 0, 0], [0.5, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0.5, 1]]
        channel_matrix: [1,2,3,4]
        readout_directions: [1,2,1,2]

.. autofunction:: ac_crosstalk

Dead time filter
================

This model applies only for the :py:class:`~pyxel.detectors.MKID` detector.

More information in :cite:p:`2020:prodhomme` section 3.3.1.

Example of the configuration file:

.. code-block:: yaml

    - name: dead_time_filter
      func: pyxel.models.readout_electronics.dead_time_filter
      enabled: true
      arguments:
        dead_time: 1.0

.. note:: This model is specific for the MKID detector.

.. autofunction:: dead_time_filter

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

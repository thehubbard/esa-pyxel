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

This model only applies to the :py:class:`~pyxel.detectors.MKID` detector.

There is a maximum limit to the achievable count rate, which is inversely proportional to the minimum distance in time between distinguishable pulse profiles: the so-called “dead time”, which is fundamentally determined by the recombination time of quasi-particles re-forming Cooper pairs :cite:p:`2020:prodhomme`.

The underlying physics of this model is described in :cite:p:`PhysRevB.104.L180506`; more information can be found on the website :cite:p:`Mazin`.

Example of the configuration file:

.. code-block:: yaml

    - name: dead_time_filter
      func: pyxel.models.readout_electronics.dead_time_filter
      enabled: true
      arguments:
        dead_time: 1.0

.. note:: This model is specific to the MKID detector.

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

Simple phase conversion
=======================

With this model you can convert :py:class:`~pyxel.data_structure.Phase`
array into :py:class:`~pyxel.data_structure.Image`, given a hard-coded multiplicative conversion factor.

Example of the configuration file:

.. code-block:: yaml

    - name: simple_phase_conversion
      func: pyxel.models.readout_electronics.simple_phase_conversion
      enabled: true

.. note:: This model is specific to the MKID detector.

.. autofunction:: simple_phase_conversion

Simple processing
=================

With this model you can convert :py:class:`~pyxel.data_structure.Signal`
array into :py:class:`~pyxel.data_structure.Image`.
User can specify optional argument ``gain_adc``. If not, the parameter from detector
:py:class:`~pyxel.detectors.Characteristics` is used.

Example of the configuration file:

.. code-block:: yaml

    - name: simple_processing
      func: pyxel.models.readout_electronics.simple_processing
      enabled: true
      arguments:
        gain_adc: 1.  # optional

.. autofunction:: simple_processing
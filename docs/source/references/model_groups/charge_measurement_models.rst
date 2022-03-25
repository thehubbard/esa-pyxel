.. _charge_measurement:

=========================
Charge Measurement models
=========================

.. currentmodule:: pyxel.models.charge_measurement
.. automodule:: pyxel.models.charge_measurement


DC offset
=========

:guilabel:`Signal` 🠆 :guilabel:`Signal`

Add a DC offset to signal array of detector.

.. code-block:: yaml

    - name: dc_offset
      func: pyxel.models.charge_measurement.dc_offset
      enabled: true
      arguments:
        offset: 0.1

.. autofunction:: dc_offset


Output pixel reset voltage APD
==============================

:guilabel:`Signal` 🠆 :guilabel:`Signal`

Add noise to signal array of detector output node using normal random distribution.

.. code-block:: yaml

    - name: output_pixel_reset_voltage
      func: pyxel.models.charge_measurement.output_pixel_reset_voltage
      enabled: true
      arguments:
        roic_drop: 3.3

.. note:: This model is specific to the :term:`APD` detector.

.. autofunction:: output_pixel_reset_voltage


Simple charge measurement
=========================

:guilabel:`Pixel` 🠆 :guilabel:`Signal`

Convert the pixels array to the signal array.

Example of the configuration file:

.. code-block:: yaml

    - name: simple_measurement
      func: pyxel.models.charge_measurement.simple_measurement
      enabled: true
      arguments:
        noise:
          - gain: 1.    # Optional

.. autofunction:: simple_measurement

Output node noise
=================

:guilabel:`Signal` 🠆 :guilabel:`Signal`

Add noise to signal array of detector output node using normal random distribution.

.. code-block:: yaml

    - name: output_noise
      func: pyxel.models.charge_measurement.output_node_noise
      enabled: true
      arguments:
        std_deviation: 1.0

.. note:: This model is specific to the :term:`CCD` detector.

.. autofunction:: output_node_noise

Output node noise CMOS
======================

:guilabel:`Signal` 🠆 :guilabel:`Signal`

Output node noise model for :term:`CMOS` detectors where readout is statistically independent for each pixel.

.. code-block:: yaml

    - name: output_noise
      func: pyxel.models.charge_measurement.output_node_noise
      enabled: true
      arguments:
        readout_noise: 1.0
        readout_noise_std: 2.0

.. note:: This model is specific to the :term:`CMOS` detector.

.. autofunction:: output_node_noise_cmos

Non-linearity
=============

:guilabel:`Signal` 🠆 :guilabel:`Signal`

With this model you can add non-linearity to :py:class:`~pyxel.data_structure.Signal` array
to simulate the non-linearity of the output node circuit.
The non-linearity is simulated by a polynomial function.
The user specifies the polynomial coefficients with the argument ``coefficients``:
a list of :math:`n` floats e.g. :math:`[a,b,c] \rightarrow S = a + bx+ cx2` (:math:`x` is signal).

Example of the configuration file where a 10% non-linearity is introduced as a function of the signal square:

.. code-block:: yaml

    - name: linearity
      func: pyxel.models.charge_measurement.output_node_linearity_poly
      enabled: true
      arguments:
        coefficients: [0, 1, 0.9]  # e- [a,b,c] -> S = a + bx+ cx2 (x is signal)

.. autofunction:: pyxel.models.charge_measurement.output_node_linearity_poly

HxRG noise generator
====================

:guilabel:`Pixel` 🠆 :guilabel:`Pixel`

With this model you can add noise to :py:class:`~pyxel.data_structure.Pixel` array,
before converting to :py:class:`~pyxel.data_structure.Signal` array in the charge measurement part of the pipeline.

It is a near-infrared :term:`CMOS` noise generator (ngHxRG) developed for the
James Webb Space Telescope (JWST) Near Infrared Spectrograph (NIRSpec)
described in :cite:p:`2015:rauscher`. It simulates many important noise
components including white read noise, residual bias drifts, pink 1/f
noise, alternating column noise and picture frame noise.

The model reproduces most of the Fourier noise
power spectrum seen in real data, and includes uncorrelated, correlated,
stationary and non-stationary noise components.
The model can simulate noise for HxRG detectors of
Teledyne Imaging Sensors with and without the SIDECAR ASIC IR array
controller.

Example of the configuration file:

.. code-block:: yaml

    - name: nghxrg
      func: pyxel.models.charge_measurement.nghxrg
      enabled: true
      arguments:
        noise:
          - ktc_bias_noise:
              ktc_noise: 1
              bias_offset: 2
              bias_amp: 2
          - white_read_noise:
              rd_noise: 1
              ref_pixel_noise_ratio: 2
          - corr_pink_noise:
              c_pink: 1.
          - uncorr_pink_noise:
              u_pink: 1.
          - acn_noise:
              acn: 1.
          - pca_zero_noise:
              pca0_amp: 1.
        window_position: [0, 0]   # Optional
        window_size: [100, 100]   # Optional
        n_output: 1
        n_row_overhead: 0
        n_frame_overhead: 0
        reverse_scan_direction: False
        reference_pixel_border_width: 4

.. autofunction:: pyxel.models.charge_measurement.nghxrg

* **kTC bias noise**
* **White readout noise**
* **Alternating column noise (ACN)**
* **Uncorrelated pink noise**
* **Correlated pink noise**
* **PCA0 noise**

.. _signal_transfer:

Signal Transfer models (CMOS)
=============================

.. important::
   This model group is only for :term:`CMOS`-based detectors!

.. currentmodule:: pyxel.models.signal_transfer
.. automodule:: pyxel.models.signal_transfer
    :members:
    :undoc-members:
    :imported-members:

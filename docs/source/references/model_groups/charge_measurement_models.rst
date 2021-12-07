.. _charge_measurement:

=========================
Charge Measurement models
=========================

.. currentmodule:: pyxel.models.charge_measurement
.. automodule:: pyxel.models.charge_measurement


Simple charge measurement
=========================

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

Add noise to signal array of detector output node using normal random distribution.

.. code-block:: yaml

    - name: output_noise
      func: pyxel.models.charge_measurement.output_node_noise
      enabled: true
      arguments:
        std_deviation: 1.0

.. autofunction:: output_node_noise

Output node noise CMOS
======================

Output node noise model for CMOS detectors where readout is statistically independent for each pixel.

.. code-block:: yaml

    - name: output_noise
      func: pyxel.models.charge_measurement.output_node_noise
      enabled: true
      arguments:
        readout_noise: 1.0
        readout_noise_std: 2.0

.. autofunction:: output_node_noise_cmos

Non-linearity
=============
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

A near-infrared CMOS noise generator (ngHxRG) developed for the
James Webb Space Telescope (JWST) Near Infrared Spectrograph (NIRSpec)
described in :cite:p:`2015:rauscher`
has been also added to the framework. It simulates many important noise
components including white read noise, residual bias drifts, pink 1/f
noise, alternating column noise and picture frame noise.

This model implemented in Python, reproduces most of the Fourier noise
power spectrum seen in real data, and includes uncorrelated, correlated,
stationary and non-stationary noise components.
The model can simulate noise for HxRG detectors of
Teledyne Imaging Sensors with and without the SIDECAR ASIC IR array
controller.

Example of the configuration file:

.. code-block:: yaml

    - name: illumination
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
        option: "elliptic"

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
   This model group is only for CMOS-based detectors!

.. currentmodule:: pyxel.models.signal_transfer
.. automodule:: pyxel.models.signal_transfer
    :members:
    :undoc-members:
    :imported-members:

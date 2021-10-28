.. _charge_measurement:

Charge Measurement models
=========================

.. currentmodule:: pyxel.models.charge_measurement
.. automodule:: pyxel.models.charge_measurement


Simple charge measurement
-------------------------

.. autofunction:: simple_measurement

Output node noise
-----------------

.. autofunction:: output_node_noise

Output node noise CMOS
----------------------

.. autofunction:: output_node_noise_cmos

HxRG noise generator
--------------------

A near-infrared CMOS noise generator (ngHxRG) developed for the
James Webb Space Telescope (JWST) Near Infrared Spectrograph (NIRSpec)
has been also added to the framework. It simulates many important noise
components including white read noise, residual bias drifts, pink 1/f
noise, alternating column noise and picture frame noise.

This model implemented in Python, reproduces most of the Fourier noise
power spectrum seen in real data, and includes uncorrelated, correlated,
stationary and non-stationary noise components.
The model can simulate noise for HxRG detectors of
Teledyne Imaging Sensors with and without the SIDECAR ASIC IR array
controller.

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

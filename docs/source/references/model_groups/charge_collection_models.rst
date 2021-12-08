.. _charge_collection:

========================
Charge Collection models
========================

.. currentmodule:: pyxel.models.charge_collection

Charge collection models are used to add to and manipulate data in :py:class:`~pyxel.data_structure.Pixel` array
inside the :py:class:`~pyxel.detectors.Detector` object.
The data represents amount of charge stored in each of the pixels.
A charge collection model is necessary to first convert from charge data stored in
:py:class:`~pyxel.data_structure.Charge` class. Multiple models are available to add detector effects after.

Simple collection
=================

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
================
This model can be used to limit the amount of charge in :py:class:`~pyxel.data_structure.Pixel` array
due to full well capacity. Values will be clipped to the value of the full well capacity.
The model uses full well capacity value specified in :py:class:`~pyxel.detectors.Characteristics` of the
:py:class:`~pyxel.detectors.Detector`, unless providing an argument ``fwc`` directly as the model argument.

Example of the configuration file:

.. code-block:: yaml

    - name: simple_full_well
      func: pyxel.models.charge_collection.simple_full_well
      enabled: true
      arguments:
          fwc: 1000  # optional

.. autofunction:: simple_full_well

Fix pattern noise
=================

Add fix pattern noise caused by pixel non-uniformity during charge collection.

.. code-block:: yaml

    - name: fix_pattern_noise
      func: pyxel.models.charge_collection.fix_pattern_noise
      enabled: true
      arguments:
          pixel_non_uniformity: filename.fits  # optional


.. autofunction:: fix_pattern_noise


Inter-pixel capacitance
=======================
This model can be used to apply inter-pixel capacitance to :py:class:`~pyxel.data_structure.Pixel` array.
When there is IPC, the signal read out on any pixel is affected by the signal in neighboring pixels.
The IPC affects the point spread function (PSF) of the optical system, modiying the shape of the objects.
More about the IPC and the math describing it can be found in :cite:p:`Kannawadi_2016`.
The amount of coupling between the pixels is described in the article by a
:math:`3\times3` matrix :math:`K_{\alpha, \alpha_+, \alpha'}`:

.. math::
    K_{\alpha, \alpha_+, \alpha'} = \begin{bmatrix}
    \alpha' & \alpha-\alpha_+ & \alpha'\\
    \alpha+\alpha_+ & 1-4(\alpha+\alpha') & \alpha+\alpha_+\\
    \alpha' & \alpha-\alpha_+ & \alpha'
    \end{bmatrix},

where :math:`\alpha` is the coupling parameter for the neighbouring pixels,
:math:`\alpha'` the coupling parameter for the pixels located on the diagonals
and :math:`\alpha_+` parameter for introducing an anisotropical coupling. In the model, the last two are optional.
The sum of the matrix elements is always 1.
The result image that is seen on the detector is a convolution of the image with the kernel matrix,
which is done using ``astropy`` convolution tools.

Example of the configuration file:

.. code-block:: yaml

    - name: simple_ipc
      func: pyxel.models.charge_collection.simple_ipc
      enabled: true
      arguments:
          coupling: 0.1
          diagonal_coupling: 0.05
          anisotropic_coupling: 0.03

.. note:: This model is specific for the CMOS detector.

.. autofunction:: simple_ipc

Simple Persistence
==================

Simple trapping / detrapping charges.

.. code-block:: yaml

    - name: simple_persistence
      func: pyxel.models.charge_collection.simple_persistence
      enabled: true
      arguments:
          trap_timeconstants: [1., 10.]      # Two traps
          trap_densities: [0.307, 0.175]

.. note:: This model is specific for the CMOS detector.

.. autofunction:: simple_persistence


Current Persistence
===================


.. autofunction:: current_persistence

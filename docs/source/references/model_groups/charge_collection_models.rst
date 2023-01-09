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


.. _charge_collection_create_store_detector:

Create and Store a detector
===========================

The models :ref:`charge_collection_save_detector` and :ref:`charge_collection_load_detector`
can be used respectively to create and to store a :py:class:`~pyxel.detectors.Detector` to/from a file.

These models can be used when you want to store or to inject a :py:class:`~pyxel.detectors.Detector`
into the current :ref:`pipeline`.

.. _charge_collection_save_detector:

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

.. _charge_collection_load_detector:

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


.. _Simple collection:

Simple collection
=================

:guilabel:`Charge` → :guilabel:`Pixel`

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

.. _Simple full well:

Simple full well
================

:guilabel:`Pixel` → :guilabel:`Pixel`

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

.. _Fixed pattern noise:

Fixed pattern noise
===================

:guilabel:`Pixel` → :guilabel:`Pixel`

With this model you can multiply :py:class:`~pyxel.data_structure.Pixel` array with
fixed pattern noise caused by pixel non-uniformity during charge collection.
User has to provide a ``filename`` or a fixed-pattern nise factor to model arguments.
Accepted file formats for the noise are ``.npy``, ``.fits``, ``.txt``, ``.data``, ``.jpg``, ``.jpeg``, ``.bmp``,
``.png`` and ``.tiff``. Use argument ``position`` to set the offset from (0,0) pixel
and set where the noise is placed onto detector. You can set preset positions with argument ``align``.
Values outside of detector shape will be cropped.
Read more about placement in the documentation of function :py:func:`~pyxel.util.fit_into_array`.
If the user provides a value for the ``fixed_pattern_noise_factor`` instead of a filename,
the model will use a simple calculation of the PRNU. In the simple calculation the ``fixed_pattern_noise_factor``
will be multiplied with the quantum_efficiency, given by the detector characteristics, and applied to the pixel array
through a lognormal distribution. The ``fixed_pattern_noise_factor`` is typically between 0.01 and 0.02 for a given sensor,
but varies from one sensor to another :cite:p:`Konnik:noises`.


Basic example of the configuration file:

.. code-block:: yaml

    - name: fixed_pattern_noise
      func: pyxel.models.charge_collection.fixed_pattern_noise
      enabled: true
      arguments:
          filename: "noise.fits"
          #fixed_pattern_noise_factor: 0.01

.. autofunction:: fixed_pattern_noise

.. _Inter pixel capacitance:

Inter-pixel capacitance
=======================

:guilabel:`Pixel` → :guilabel:`Pixel`

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

.. note:: This model is specific for the :term:`CMOS` detector.

.. autofunction:: simple_ipc

.. _Simple persistence:

Simple Persistence
==================

:guilabel:`Pixel` → :guilabel:`Pixel`

With this model you can simulate the effect of persistence changing :py:class:`~pyxel.data_structure.Pixel` array.
The simple model takes as input a list of trap time constants together with a list of trap densities
and assuming the trap densities are uniform over the whole detector area.
Additionally user can also specify trap full well capacities using the ``trap_capacities`` parameter.
At each iteration of the pipeline, the model  will compute the amount of trapped charges in this iteration, add it
to the memory of the detector and then remove this amount from the pixel array.
More on the persistence model can be found in  :cite:p:`2019:persistence`.

Example of the configuration file:

.. code-block:: yaml

    - name: simple_persistence
      func: pyxel.models.charge_collection.simple_persistence
      enabled: true
      arguments:
        trap_time_constants: [1., 10.]  # Two different traps
        trap_densities: [0.307, 0.175]
        trap_capacities: [100., 100.]  # optional


.. note:: This model is specific for the :term:`CMOS` detector.

.. autofunction:: simple_persistence

.. _Persistence:

Persistence
===========

:guilabel:`Pixel` → :guilabel:`Pixel`

With this model you can simulate the effect of persistence changing :py:class:`~pyxel.data_structure.Pixel` array.
The more advanced model takes as input a list of trap time constants together with a list of trap proportions.
For trap densities user has to provide a 2D map of densities.
This model assumes trap density distribution over the detector area is the same for all traps
and the trap densities are computed using the map and trap proportions.
Additionally user can also specify trap full well capacity map.
At each iteration of the pipeline, the model  will compute the amount of trapped charges in this iteration, add it
to the memory of the detector and then remove this amount from the pixel array.
More on the persistence model can be found in  :cite:p:`2019:persistence`.

Use arguments ``trap_densities_position`` and ``trap_capacities_position`` to set the maps offset from (0,0) pixel
and set where the input map is placed onto detector.
You can set preset positions with arguments ``trap_densities_align`` and ``trap_capacities_align``.
Values outside of detector shape will be cropped.
Read more about placement in the documentation of function :py:func:`~pyxel.util.fit_into_array`.

Example of the configuration file:

.. code-block:: yaml

    - name: persistence
      func: pyxel.models.charge_collection.persistence
      enabled: true
      arguments:
        trap_time_constants: [1, 10, 100, 1000, 10000]
        trap_proportions: [0.307, 0.175, 0.188, 0.136, 0.194]
        trap_densities_filename: trap_densities.fits
        trap_capacities_filename: trap_capacities.fits  # optional

.. note:: This model is specific for the CMOS detector.

.. autofunction:: persistence

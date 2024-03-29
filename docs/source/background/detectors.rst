.. _detectors:

=========
Detectors
=========

The :py:class:`~pyxel.detectors.Detector` object serves as the primary input for the detection pipeline,
functioning as a repository for all data required by the models within the :ref:`pipeline`.
Note: Not all models are compatible with every detector type. To verify compatibility, consult the model table and
refer to the :ref:`model_groups_explanation` within the comprehensive list of all models provided (see :ref:`models`).
This will ensure that the model you intend to use is suitable for the selected detector type.
The detector object traverses through all the encompassing models, represented by model functions.
As per the specifications outlined in the ``YAML`` configuration file, a single :py:class:`~pyxel.detectors.Detector`
object is instantiated for each exposure.

.. _detector_properties:

Detector properties
-------------------

The :py:class:`~pyxel.detectors.Detector` object encompasses all information related to the physical attributes of the
simulated detector.
These detector properties can be classified into the following categories:
:py:class:`~pyxel.detectors.Geometry`, :py:class:`~pyxel.detectors.Characteristics`,
and :py:class:`~pyxel.detectors.Environment`, as shown in the image.
:ref:`time_properties` and the
``rows`` and ``columns`` of :py:class:`~pyxel.detectors.Geometry` are mandatory, as highlighted in the image.
These properties are utilized by multiple models within the pipeline and remain constant throughout a pipeline run.
Notably, the category of detector properties labeled ``Material``, was temporarily removed in version
1.0 due to its lack of utilization.

.. figure:: _static/detector.png
    :width: 600px
    :alt: detector
    :align: center


.. _data_structure:

Detector data containers
------------------------
The detector additionally contains data buckets that store simulated data, including
input photon distribution (photons), number of charge carriers generated (carrier type), signal variation [#]_ in pixels
(voltage, phase), and digitised image value (ADU).
The data buckets are not initialized before running a pipeline. The models inside the model groups must initialize
the data buckets.
These data containers undergo modifications by the models within the
pipeline, ultimately altering the state of the output detector upon completion of the pipeline.
The data structures involved are:
:py:class:`~pyxel.data_structure.Scene`, :py:class:`~pyxel.data_structure.Photon`,
:py:class:`~pyxel.data_structure.Charge`, :py:class:`~pyxel.data_structure.Pixel`,
:py:class:`~pyxel.data_structure.Signal`, :py:class:`~pyxel.data_structure.Image`
and :py:class:`~pyxel.data_structure.Phase` class as shown in the image below.
:py:class:`~pyxel.data_structure.Scene` is converted to multi-wavelength (photon/nm) or monochromatic (photon)
:py:class:`~pyxel.data_structure.Photon` depending which models are used.
For a MKID type detector, the :py:class:`~pyxel.data_structure.Pixel` is converted to
:py:class:`~pyxel.data_structure.Image` over the :py:class:`~pyxel.data_structure.Phase` container.

The classes are storing the data values either inside a Pandas
:py:class:`pandas.DataFrame` or in a NumPy :py:class:`numpy.ndarray`. Via DataFrame or
NumPy array handling functions, models can modify properties of photons,
charges, etc., like position, kinetic energy, number of electrons per charge packet,
signal amplitude, etc.

.. figure:: _static/data.png
    :width: 800px
    :alt: detector
    :align: center


.. [#] Which is going to be a phase shift, in the case of MKIDs---once their underlying physics is fully implemented.

.. _time_properties:

Time properties
---------------

As shown in the image, the :py:class:`~pyxel.detectors.Detector` object also tracks time.
There are multiple properties inside the :py:class:`~pyxel.detectors.Detector` object:
``time`` is the time since ``start_time`` (which can be different to 0), ``absolute time`` is the time since 0,
and ``time_step`` is the time since last readout. Those properties can be used by the time-sensitive models.

.. _detector_types:

Implemented detector types:
---------------------------

* :ref:`CCD architecture`
* :ref:`CMOS architecture`
* :ref:`MKID architecture`
* :ref:`APD architecture`

.. toctree::
   :caption: Detector types
   :maxdepth: 1
   :hidden:

   detectors/ccd.rst
   detectors/cmos.rst
   detectors/mkid.rst
   detectors/apd.rst

.. _detectors:

=========
Detectors
=========

The :py:class:`~pyxel.detectors.Detector` object is the main input of the detection pipeline.
Therefore it is a container for all the data that the models need access to inside the :ref:`pipeline`.
It is passed through all the including models represented by model functions.
According to the ``YAML`` configuration file, one :py:class:`~pyxel.detectors.Detector` object is
instantiated for each exposure.

.. figure:: _static/detector.png
    :width: 600px
    :alt: detector
    :align: center

.. _detector_properties:

Detector properties
-------------------

The initial group of data buckets within the :py:class:`~pyxel.detectors.Detector` object encompasses all information
and data pertinent to the physical attributes of the simulated detector.
These can be classified into the following categories:
:py:class:`~pyxel.detectors.Geometry`, :py:class:`~pyxel.detectors.Characteristics`,
and :py:class:`~pyxel.detectors.Environment`, as shown in the image.
:ref:`time_properties` and the
``row`` and ``col`` of :py:class:`~pyxel.detectors.Geometry` are mandatory, as highlighted in the image.
These properties are utilized by multiple models within the pipeline and remain constant throughout a pipeline run,
though subject to variation depending on the specific detector employed.
Notably, the category of detector properties labeled ``Material``, was temporarily removed in version
1.0 due to its lack of utilization.

.. _data_structure:

Data
====

The detector also holds data buckets storing the simulated data, such as
input photon distribution (photons), number of charge carriers generated (carrier type), signal variation [#]_ in pixels
(voltage, phase), and digitised image value (ADU). The data buckets are modified by the models in the pipeline
and the state of the output detector at the end of the pipeline is changed.
These data structures are:
:py:class:`~pyxel.data_structure.Scene`, :py:class:`~pyxel.data_structure.Photon`,
:py:class:`~pyxel.data_structure.Charge`, :py:class:`~pyxel.data_structure.Pixel`,
:py:class:`~pyxel.data_structure.Signal`, :py:class:`~pyxel.data_structure.Image`
and :py:class:`~pyxel.data_structure.Phase` class.

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

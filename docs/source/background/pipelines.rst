.. _pipelines:

#########
Pipelines
#########

The core algorithm of the architecture is the Detection pipeline allowing to
host any type of :ref:`models <models_explanation>` in an arbitrary number.
It is an instance of :py:class:`~pyxel.pipelines.DetectionPipeline` class.

Inside the pipeline the :ref:`models <models_explanation>` are grouped into 7 different
levels per detector type imitating the working principle of the detector, for example
in case of a CCD the model levels are :ref:`photon generation <photon_generation>`,
:ref:`optics <optical>`, :ref:`charge <charge_transfer>`, :ref:`generation <charge_generation>`,
:ref:`charge collection <charge_collection>`, :ref:`charge transfer <charge_transfer>`,
:ref:`charge measurement <charge_measurement>` and :ref:`readout electronics <readout_electronics>`
in this order.

Each level is based on a
for loop, looping over all the included and selected models in a predefined
order, which can be changed by the user. All the models in a pipeline, get
and modify the same :py:class:`~pyxel.detectors.Detector` object one after another.
At the end, the pipeline returns the :py:class:`~pyxel.detectors.Detector` object
as an output ready to generate output files from results.

.. image:: _static/pyxel_ccd_pipeline.png
    :scale: 20%
    :alt: ccd_pipeline
    :align: center

.. _models_explanation:

Models
======

By models, we mean various analytical functions, numerical methods or
algorithms implemented in order to approximate, calculate, visualize
electro-optical performance and degradation due to the operational
environment (space, laboratory test) and its effects (e.g. radiation
damage).

Models are Python functions with a :py:class:`~pyxel.detectors.Detector` object
defined as their input argument. The model function has to be
added to the ``YAML`` configuration file.
Then the function is automatically called by Pyxel inside a loop of its
model group and the :py:class:`~pyxel.detectors.Detector` object is passed to it.
The model may modifies this :py:class:`~pyxel.detectors.Detector` object which is
also used and modified further by the next models in the pipeline.


.. _model_groups_explanation:

Model groups
------------

Models are grouped into 7 model levels per detector type according to
which object of the :py:class:`~pyxel.detectors.Detector` object is used or modified by
the models. These levels correspond roughly to the detector fundamental
functions.

Models in Pyxel makes changes and storing there data in data structure
classes (:py:class:`~pyxel.data_structure.Photon`, :py:class:`~pyxel.data_structure.Charge`,
:py:class:`~pyxel.data_structure.Pixel`, :py:class:`~pyxel.data_structure.Signal` or
:py:class:`~pyxel.data_structure.Image` class).
For details, see the :ref:`Data Structure <data_structure>` page.

Models could also modify any detector attributes (like Quantum Efficiency,
gains, temperature, etc.) stored in a Detector subclass
(:py:class:`~pyxel.detectors.Characteristics`, :py:class:`~pyxel.detectors.Environment`,
:py:class:`~pyxel.detectors.Material`).


Detector attributes changes could happen globally (on detector level)
or locally (on pixel level or only for a specific detector area).

.. figure:: _static/model-table.png
    :scale: 70%
    :alt: models
    :align: center

All the 8 model levels, which are imitating the physical working principles of imaging detectors. They were
grouped according to which physics data storing objects are modified by them. Note that 2 out of the 8 levels are
specific to a single detector type.


Model inputs
------------

Models functions have at least one compulsory input argument,
which is either a general, a :py:class:`~pyxel.detectors.CCD` or
a :py:class:`~pyxel.detectors.CMOS` type :py:class:`~pyxel.detectors.Detector` object,
depending on what the model is supposed to simulate:
a general (e.g. cosmic rays),
a CCD (e.g. CTI) or a CMOS (e.g. Alternating Column Noise) specific
detector effect.

Any other (optional) input arguments can be defined for the model as well,
which will be loaded from the :ref:`YAML <yaml>` file automatically.

Adding a new model
------------------

Users and developers can easily add any kind of new or already existing
model to Pyxel, thanks to the easy-to-use model plug-in mechanism
developed for this purpose.

For more details, see the :ref:`Adding new models <new_model>` page.

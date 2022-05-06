.. _yaml:

===================
Configuration files
===================

The framework uses a user-friendly, structured ``YAML`` configuration file as an
input, which defines the running mode, the detector properties, detector effect models and
their input arguments.
The configuration file is loaded with the function :py:func:`~pyxel.load`

Structure
=========

After the framework loads and validates the ``YAML`` file (with .yaml ot .yml extension),
it creates :py:class:`~pyxel.detectors.Detector` and
:py:class:`~pyxel.pipelines.DetectionPipeline` object(s) based on
the ``YAML`` file with all the information needed for the framework to run
the simulation.

The ``YAML`` configuration file of Pyxel is structured
similarly to the architecture, so the Pyxel class hierarchy can be
recognized in the group hierarchy of ``YAML`` files.

The groups and subgroups of the ``YAML`` file create objects from the
classes defined with their *class* arguments. During this process,
classes get all the parameters as input arguments defined within the group
or subgroup.

Running mode
------------

In the beginning of the configuration file, the user should define
the running mode. This can be :ref:`exposure <exposure_mode>`,
:ref:`observation <observation_mode>`, :ref:`calibration <calibration_mode>`.
For details, see :ref:`running_modes`.

Detector
--------

All arguments of Detector subclasses (:py:class:`~pyxel.detectors.Geometry`,
:py:class:`~pyxel.detectors.Characteristics`, :py:class:`~pyxel.detectors.Environment`,
:py:class:`~pyxel.detectors.Material` and :py:class:`~pyxel.detectors.Optics`) are defined here.
For details, see :ref:`detectors`.

Pipeline
--------

It contains the model levels as subgroups
(*photon_generation*, *optics*, *charge_generation*, etc.).
For details, see :ref:`pipeline`.

The order of model levels and models are important,
as the execution order is defined here!

* **photon_generation**

* **optics**

* **charge_generation**

* **charge_collection**

* **(charge_transfer)**

* **charge_measurement**

* **(signal_transfer)**

* **readout_electronics**


Models need a *name* which defines the path to the model wrapper
function. Models also have an *enabled* boolean switch, where the user
can enable or disable the given model. The optional and compulsory
arguments of the model functions have to be listed inside the
*arguments*. For details, see :ref:`models`.

YAML basic syntax
=================

A quick overview of possible inputs and structures in the YAML file.

**Numbers**

.. code-block:: yaml

    one:  1.
    two:   3.e-6
    three:  10


**Strings**

.. code-block:: yaml

    string: foo
    forced_string: "bar"

**Lists**

.. code-block:: yaml

    list: [1,2]

    or

    list:
      - 1
      - 2

**Dictionaries**

.. code-block:: yaml

    dictionary: {"foo":1, "bar":2}

    or

    dictionary:
      foo: 1
      bar: 2

**Comments**

.. code-block:: yaml

    # just a comment

**Example**

.. code-block:: yaml

    foo:
      - 1
      - 2
    bar:
      one:
        - alpha
        - "beta"
      two: 5.e-3

    would be converted to

    {"foo":[1,2], "bar":{'one':["alpha", "beta"], "two":5.e-3}}


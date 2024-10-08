.. _yaml:

===================
Configuration files
===================

The framework uses a structured ``YAML`` configuration file as an
input, which defines the running mode, the detector properties, detector effect models and
their input arguments.
The configuration file is loaded with the function :py:func:`~pyxel.load`.

Despite the configuration file being human-readable and easy to understand,
it is still possible to make mistakes that result in errors during the simulation.
Therefore a configuration file validation process based on JSON schema can be used
to further improve the user experience. More information here: :ref:`json_schema`.

Structure
=========

The file consists of three separate parts, each representing a class in the Pyxel architecture.
They define the running mode, the detector properties, and the pipeline - the models the user wants to apply.
When the YAML configuration file is loaded, the nested dictionaries, lists, numbers,
and strings are used to directly initialize the Pyxel classes. See examples below.

The ``YAML`` configuration file of Pyxel is structured
similarly to the architecture, so the Pyxel class hierarchy can be
recognized in the group hierarchy of ``YAML`` files.

Running mode
------------

In the beginning of the configuration file, the user should define
the running mode. This can be :ref:`exposure <exposure_mode>`,
:ref:`observation <observation_mode>`, :ref:`calibration <calibration_mode>`.
For details, see :ref:`running_modes`. Example for exposure mode can be seen below.

.. code-block:: yaml

    exposure:
      working_directory: my_folder   # This parameter is optional

      readout:
        times: [1., 5., 7.]
        non_destructive: false

      outputs:
        output_folder: "output"
        save_data_to_file:
          - detector.image.array: ["fits"]
        save_exposure_data:
          - dataset: ["nc"]

.. note::
    The **optional** parameter ``working_directory`` is used to define the current working directory.
    By default, the current working directory is directory where the YAML file is located.

    This ``working_directory`` will be used as the parent directory for **all** relative paths
    defined in the YAML file.

    Example:

    .. code-block:: yaml

        working_directory: ~/my_folder     # <== define working directory to `~/my_folder` (optional)
        simulation:
          mode: calibration
          calibration:
            target_data_path: ['CTI/input/data.fits']  # <== will be converted as `~/my_folder/CTI/input/data.fits`
                                +-----------------+                                +---------+
                                        |                                               |
                                    relative path                            from 'working_directory'


Detector
--------

All arguments of Detector subclasses (:py:class:`~pyxel.detectors.Geometry`,
:py:class:`~pyxel.detectors.Characteristics`, :py:class:`~pyxel.detectors.Environment`) are defined here.
Since version 2.0, Pyxel supports multi-wavelength functionality.
In addition to providing the wavelength as input for models capable of handling multiple wavelengths,
users can also specify wavelength information within the detector object's environment.
This can involve setting a single value for monochromatic wavelength handling or specifying parameters such as
``cut_on``, ``cut_off`` and ``resolution`` to define the wavelength range and resolution for creating a multi-wavelength
detector object.

Example of a monochromatic detector object:

.. code-block:: yaml

    ccd_detector:

      geometry:

        row: 512
        col: 512
        total_thickness: 40.
        pixel_vert_size: 15.
        pixel_horz_size: 15.
        pixel_scale: 1.38

      environment:
        temperature: 80
        wavelength: 600

      characteristics:
        quantum_efficiency: 1.
        charge_to_volt_conversion: 5.e-6
        pre_amplification: 5.
        adc_bit_resolution: 16
        adc_voltage_range: [0.,5.]
        full_well_capacity: 90000

Example of a multi-wavelength detector object:

.. code-block:: yaml

    ccd_detector:

      geometry:

        row: 512
        col: 512
        total_thickness: 40.
        pixel_vert_size: 15.
        pixel_horz_size: 15.
        pixel_scale: 1.38

      environment:
        temperature: 80
        wavelength:
          cut_on: 550
          cut_off: 650
          resolution: 10

      characteristics:
        quantum_efficiency: 1.
        charge_to_volt_conversion: 5.e-6
        pre_amplification: 5.
        adc_bit_resolution: 16
        adc_voltage_range: [0.,5.]
        full_well_capacity: 90000

For more details on the :py:class:`~pyxel.detectors.Detector` object, see also :ref:`detectors`.


Pipeline
--------

The pipeline contains the model functions grouped into model groups
(*scene_generation*, *photon_collection*, *charge_generation*, etc.).
For more details, see :ref:`pipeline`.

The order of model levels and models are important,
as the execution order is defined here!

* :ref:`scene_generation`

* :ref:`photon_collection`

* :ref:`charge_generation`

* :ref:`charge_collection`

* (:ref:`phasing`)

* (:ref:`charge_transfer`)

* :ref:`charge_measurement`

* :ref:`readout_electronics`

* :ref:`data_processing`

Models need a ``name`` which defines the path to the model wrapper
function. Models also have an ``enabled`` boolean switch, where the user
can enable or disable the given model. The optional and compulsory
arguments of the model functions have to be listed inside the
``arguments``.
For more details, see :ref:`models`.

.. code-block:: yaml

    pipeline:

      # -> photon
      photon_collection:

        - name: illumination
          func: pyxel.models.photon_collection.illumination
          enabled: true
          arguments:
              level: 100.
              time_scale: 1.

        - name: shot_noise
          func: pyxel.models.photon_collection.shot_noise
          enabled: true

      # photon -> charge
      charge_generation:
        - name: photoelectrons
          func: pyxel.models.charge_generation.simple_conversion
          enabled: true

   ...

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


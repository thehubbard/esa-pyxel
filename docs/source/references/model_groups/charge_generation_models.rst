.. _charge_generation:

========================
Charge Generation models
========================

.. currentmodule:: pyxel.models.charge_generation

Charge generation models are used to add to and manipulate data in :py:class:`~pyxel.data_structure.Charge` array
inside the :py:class:`~pyxel.detectors.Detector` object.
The values in the :py:class:`~pyxel.data_structure.Charge` array represent charge in electron.
If the :ref:`photon collection <photon_collection>` model group
is used, a model like :ref:`Simple photoconversion` needs to be enabled in the pipeline to make the conversion from
:guilabel:`Photon` to :guilabel:`Charge`.
Otherwise, a model like :ref:`Load charge` needs to be enabled to initialize the
:py:class:`~pyxel.data_structure.Charge` array.


.. _charge_generation_create_store_detector:

Create and Store a detector
===========================

The models :ref:`charge_generation_save_detector` and :ref:`charge_generation_load_detector`
can be used respectively to create and to store a :py:class:`~pyxel.detectors.Detector` to/from a file.

These models can be used when you want to store or to inject a :py:class:`~pyxel.detectors.Detector`
into the current :ref:`pipeline`.

.. _charge_generation_save_detector:

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


.. _charge_generation_load_detector:

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


.. _Simple photoconversion:

Simple photoconversion
======================

:guilabel:`Photon` → :guilabel:`Charge`

With this model you can create and add charge to :py:class:`~pyxel.detectors.Detector` via photoelectric effect
by converting photons to charge.
This model supports both monochromatic and multiwavelength photons, converting either a 2D photon array or 3D photon array
to the 2D charge array.
If the previous model group :ref:`photon collection <photon_collection>` returns a 3D photon array, the
photon array will be integrated along the wavelength dimension before applying the quantum efficiency (:term:`QE`).

Binomial sampling of incoming Poisson distributed photons is used in the conversion by default,
with probability :term:`QE`. It can be turned off by setting the argument ``binomial_sampling`` to ``False``.
User can provide an optional quantum efficiency (``quantum_efficiency``) parameter.
If not provided, quantum efficiency from detector :py:class:`~pyxel.detectors.Characteristics` is used.
It is also possible to set the seed of the random generator with the argument ``seed``.

Basic example of YAML configuration model:

.. code-block:: yaml

    - name: simple_conversion
      func: pyxel.models.charge_generation.simple_conversion
      enabled: true
      arguments:
        quantum_efficiency: 0.8  # optional

.. autofunction:: simple_conversion

.. warning:: Model assumes shot noise model was applied to photon array when using binomial sampling.

.. _Conversion with custom QE map:

Conversion with custom QE map
=============================

:guilabel:`Photon` → :guilabel:`Charge`

With this model you can create and add charge to :py:class:`~pyxel.detectors.Detector` via photoelectric effect
by converting photons in charge.
Binomial sampling of incoming Poisson distributed photons is used in the conversion by default,
with probability :term:`QE`. It can be turned off by setting the argument ``binomial_sampling`` to ``False``.
Besides that, user can input a custom quantum efficiency map by providing a ``filename`` of the :term:`QE` map.
Accepted file formats for :term:`QE` map are ``.npy``, ``.fits``, ``.txt``, ``.data``, ``.jpg``, ``.jpeg``, ``.bmp``,
``.png`` and ``.tiff``. Use argument ``position`` to set the offset from (0,0) pixel
and set where the input :term:`QE` map is placed onto detector. You can set preset positions with argument ``align``.
Values outside of detector shape will be cropped.
Read more about placement in the documentation of function :py:func:`~pyxel.util.fit_into_array`.

Basic example of YAML configuration model:

.. code-block:: yaml

    - name: conversion_with_qe_map
      func: pyxel.models.charge_generation.conversion_with_qe_map
      enabled: true
      arguments:
        filename: data/qe_map.npy

.. note::
    You can find an example of this model used in this Jupyter Notebook
    :external+pyxel_data:doc:`use_cases/HxRG/h2rg`
    from `Pyxel Data <https://esa.gitlab.io/pyxel-data>`_.

.. autofunction:: conversion_with_qe_map

.. warning:: Model assumes shot noise model was applied to photon array when using binomial sampling.


.. _Apply QE curve:

Apply QE curve
==============

.. note::
    This model operates multi-wavelength photons.

:guilabel:`Photon` → :guilabel:`Charge`

With this model you can create and add charge to :py:class:`~pyxel.detectors.Detector` via photoelectric effect
by converting photons in charge. Loading QE vs wavelength values from a file to apply the QE to the photon array.
Accepted file formats are ``.npy``, ``.fits``, ``.txt``, ``.data`` and ``.csv``.
The column containing wavelength information should be in nanometers.
After the photoconversion from photon to charge, applying the QE values to the photon array
takes places and finally integrating along the wavelength dimension to get a 2D charge array as output.

Basic example of YAML configuration model:

.. code-block:: yaml

    - name: load_qe_curve
      func: pyxel.models.charge_generation.apply_qe_curve
      enabled: false
      arguments:
        filename: "qe_curve.csv"
        wavelength_col_name: "corrected lambda / nm"
        qe_col_name: "QE"

.. autofunction:: apply_qe_curve

.. _Conversion with 3D QE map:

Conversion with 3D QE map
=========================

.. note::
    This model operates multi-wavelength photons.


:guilabel:`Photon` → :guilabel:`Charge`

With this model you can create and add charge to :py:class:`~pyxel.detectors.Detector` via photoelectric effect
by converting photons in charge. Loading QE values from a file to apply the QE to the photon array.
Loading a 3D QE map from a file containing one QE map in the size of the detector per wavelength to apply the QE
to the photon array. The file format must be netCDF, so ending with ``.nc`` to be able to read in.
The file loaded will be interpreted as :py:class:`xarray.DataArray` and should have the "wavelength" as coordinate, such that the
wavelength resolution of the QE map data can be interpolated to match to the resolution of the wavelength used in
the photon array. After that the photoconversion from photon to charge, applying the QE values to the photon array
takes places and finally integrating along the wavelength dimension to get a 2D charge array as output.

Basic example of YAML configuration model:

.. code-block:: yaml

    - name: conversion_with_3d_qe_map
      func: pyxel.models.charge_generation.conversion_with_3d_qe_map
      enabled: true
      arguments:
        filename: "qe_map.nc"

.. autofunction:: exponential_qe

.. _Exponential absorption law :

Exponential absorption law
==========================

.. note::
    This model can operate with monochromatic and multi-wavelength photons.

.. note::
    This model is valid only for silicon detectors at the moment.


:guilabel:`Photon` → :guilabel:`Charge`

With this model you can create and add charge to :py:class:`~pyxel.detectors.Detector` via photoelectric effect by estimating the
QE of your detector through an exponential absorption law (:cite:p:'2001:janesick').

The QE calculation depends on the detector type (**Back-Illuminated** or **Front-Illuminated**) and incorporates
various parameters. The equations are as follows:

1. **Back-Illuminated (BI) Detector**

The QE for a Back-Illuminated detector is given by:

.. math::

   QE = CCE \cdot (1 - R) \cdot \left(1 - e^{- \frac{x_{\text{epi}}}{\alpha}} \right)

Where:
- :math:`CCE`: Charge collection efficiency (fraction, between 0 and 1)
- :math:`R`: Reflectivity (fraction, between 0 and 1)
- :math:`x_{\text{epi}}`: Thickness of the epitaxial layer (in cm)
- :math:`\alpha`: Absorptivity (in :math:`\text{cm}^{-1}`)

2. **Front-Illuminated (FI) Detector**

For a Front-Illuminated detector, the QE calculation includes the additional effect of the poly layer. The formula is:

.. math::

   QE = CCE \cdot (1 - R) \cdot e^{- \frac{x_{\text{poly}}}{\alpha}} \cdot \left(1 - e^{- \frac{x_{\text{epi}}}{\alpha}} \right)

Where:
- :math:`x_{\text{poly}}`: Thickness of the poly layer (in cm)
- Other terms are as defined above.

**Temperature-Dependent Absorptivity Correction**

The absorptivity, :math:`\alpha`, is corrected for temperature changes using the following formula:

.. math::

   \alpha' = \alpha \cdot e^{c \cdot \Delta T}

Where:
- :math:`\alpha'`: Adjusted absorptivity at the new temperature.
- :math:`\alpha`: Absorptivity at the reference temperature (300 K).
- :math:`c`: Temperature correction coefficient (in :math:`1/\text{K}`).
- :math:`\Delta T`: Temperature difference from the reference temperature (in :math:`\text{K}`).

The embedded conversion coefficient :math:`c` is wavelength and temperature-specific.

"""
2. ** User-specified coefficients **

In case you want to operate with your own conversion coefficients, you can add an additional column to the ``.csv`` file, with a :math:`c` value for each working wavelength.

.. warning::
    The reference model has been validated at 300 K; the conversion equation may be inaccurate for coefficients measured at different temperatures.

Once the QE is computed for the required wavelength(s) and at the desired temperature, it is applied to the detector photon array and a charge array is generated for the
next steps of the pipeline.

Basic example of YAML configuration model for monochromatic pipeline:

.. code-block:: yaml

    - name: exponential_qe
      func: pyxel.models.charge_generation.exponential_qe
      enabled: true
      arguments:
          filename: qe_cleaned_data.csv
          x_epi: 0.0002
          detector_type: "BI" #or "FI"
          default_wavelength: 750.0 #in nm
          delta_t: -200.0 #optional, from the reference 300K
          x_poly: 0.0001 #only in case of "FI" detector
.. autofunction:: exponential_qe

Basic example of YAML configuration model for multi-wavelength pipeline:

.. code-block:: yaml

    - name: exponential_qe
      func: pyxel.models.charge_generation.exponential_qe
      enabled: true
      arguments:
          filename: qe_cleaned_data.csv
          x_epi: 0.0002
          detector_type: "BI" #or "FI"
          default_wavelength: 'multi'
          delta_t: -200.0 #optional, from the reference 300K
          x_poly: 0.0001 #only in case of "FI" detector
.. autofunction:: exponential_qe
.. _Load charge:

Load charge
===========

:guilabel:`Charge` → :guilabel:`Charge`

With this model you can add charge to :py:class:`~pyxel.detectors.Detector` by loading charge values from a file.
Accepted file formats are ``.npy``, ``.fits``, ``.txt``, ``.data``, ``.jpg``, ``.jpeg``, ``.bmp``,
``.png`` and ``.tiff``. Use argument ``position`` to set the offset from (0,0) pixel
and set where the input charge is placed onto detector. You can set preset positions with argument ``align``.
Values outside of detector shape will be cropped.
Read more about placement in the documentation of function :py:func:`~pyxel.util.fit_into_array`.
Use argument ``time_scale`` to set the time scale of the input charge, default is 1 second.

Basic example of YAML configuration model:

.. code-block:: yaml

    - name: load_charge
      func: pyxel.models.charge_generation.load_charge
      enabled: true
      arguments:
        filename: data/charge.npy
        position: [0,0]

.. autofunction:: load_charge

.. _Charge injection:

Charge injection
================

.. note:: This model is specific for the :term:`CCD` detector.

:guilabel:`Charge` → :guilabel:`Charge`

With this model you can inject arbitrary charge block into rows of a :py:class:`~pyxel.detectors.CCD` detector.
Charge will be injected uniformly from row number ``block_start`` to row number ``block_end``.

Example of YAML configuration model:

.. code-block:: yaml

    - name: charge_blocks
      func: pyxel.models.charge_generation.charge_blocks
      enabled: true
      arguments:
        charge_level: 100
        block_start: 10
        block_end: 50

.. autofunction:: charge_blocks

.. _Charge deposition model:

Charge deposition model
=======================

:guilabel:`Charge` → :guilabel:`Charge`

With this model it is possible to simulate the deposition of charge in the detector by ionized particles using user-provided stopping power curves.
It is possible to simulate mono-energetic beams (with a certain spread in energy) or provide an energy distribution (e.g., representative of the radiation environment).
Stopping power curves for protons in silicon and for protons in MCT alloy are provided. Similarly, the proton energy distribution at L2 with and without 11-mm aluminium shielding is provided within Pyxel.
This model is not as realistic as CosmiX but it is faster and easier to apply to a wide range of material and particles.
In particular due to its simplistic nature, it fails at reproducing the deposition of only a small amount of charge.

Example of the configuration file:


.. code-block:: yaml

    - name: charge_deposition
      func: pyxel.models.charge_generation.charge_deposition
      enabled: true
      arguments:
        flux: 100
        step_size: 1.
        energy_mean: 1.
        energy_spread: .1
        energy_spectrum: data/proton_L2_solarMax_NoShielding.txt
        energy_spectrum_sampling: log
        ehpair_creation: 3.6
        material_density: 2.33
        particle_direction: isotropic
        stopping_power_curve: data/protons-in-silicon_stopping-power.csv

.. autofunction:: pyxel.models.charge_generation.charge_deposition


.. _Charge deposition model in MCT:

Charge deposition model in MCT
==============================

:guilabel:`Charge` → :guilabel:`Charge`

This model is the same as charge deposition model but is specific to MCT material. It computes the e-h pair creation (assuming it is 3 times the bandgap) and the alloy density based on the detector temperature and cut-off wavelength.

Example of the configuration file:


.. code-block:: yaml

    - name: charge_deposition
      func: pyxel.models.charge_generation.charge_deposition_in_mct
      enabled: true
      arguments:
        flux: 100
        step_size: 1.
        energy_mean: 1.
        energy_spread: .1
        energy_spectrum: data/proton_L2_solarMax_NoShielding.txt
        energy_spectrum_sampling: log
        cutoff_wavelength: 2.5
        particle_direction: isotropic
        stopping_power_curve: data/mct-stopping-power.csv
        
.. autofunction:: pyxel.models.charge_generation.charge_deposition_in_mct

.. _CosmiX cosmic ray model:

CosmiX cosmic ray model
=======================

:guilabel:`Charge` → :guilabel:`Charge`

A cosmic ray event simulator was the first model added to Pyxel.
Initially it was a simple, semi-analytical model in Fortran using the stopping
power curve of protons to optimize the on-board source detection algorithm
of the Gaia telescope to discriminate between stars and cosmic rays. Then it
was reimplemented in Python as TARS (Tools for Astronomical Radiation
Simulations) and later as CosmiX.

With this model you can add the effect of cosmic rays to the :py:class:`~pyxel.data_structure.Charge` data structure.
See the documentation below for descriptions of parameters.
CosmiX model is described in detail in :cite:p:`2020:cosmix`.

* Developed by: David Lucsanyi, ESA

.. figure:: _static/cosmix.png
    :scale: 50%
    :alt: Poppy
    :align: center

    CosmiX cosmix ray model


Example of the configuration file using default ``running_mode: stepsize`` with step size files,
all with the same incident energy of 100 MeV and for 5 different thicknesses of 40 µm, 50 µm, 60 µm, 70 µm and 100 µm.

.. code-block:: yaml

    - name: cosmix
      func: pyxel.models.charge_generation.cosmix
      enabled: true
      arguments:
        simulation_mode: cosmic_ray
        running_mode: "stepsize"
        particle_type: proton
        initial_energy: 100.          # MeV
        particles_per_second: 100
        incident_angles:
        starting_position:
        spectrum_file: 'data/proton_L2_solarMax_11mm_Shielding.txt'
        seed: 4321

.. note::
    You can find examples of this model in these Jupyter Notebooks from `Pyxel Data <https://esa.gitlab.io/pyxel-data>`_:

    * :external+pyxel_data:doc:`use_cases/CCD/ccd`
    * :external+pyxel_data:doc:`use_cases/CMOS/cmos`
    * :external+pyxel_data:doc:`use_cases/HxRG/h2rg`


Another example of the configuration file using default ``running_mode: stepsize`` with defined step size files.

.. code-block:: yaml

    - name: cosmix
      func: pyxel.models.charge_generation.cosmix
      enabled: true
      arguments:
        simulation_mode: cosmic_ray
        running_mode: "stepsize"
        particle_type: proton
        initial_energy: 100.          # MeV
        particles_per_second: 100
        incident_angles:
        starting_position:
        spectrum_file: 'data/proton_L2_solarMax_11mm_Shielding.txt'
        seed: 4321
        stepsize:
          - type: proton
            energy:    100.0  # MeV
            thickness: 40.0   # um
            filename:  pyxel/models/charge_generation/cosmix/data/stepsize_proton_100MeV_40um_Si_10k.ascii
          - type: proton
            energy:    100.0  # MeV
            thickness: 50.0   # um
            filename:  pyxel/models/charge_generation/cosmix/data/stepsize_proton_100MeV_50um_Si_10k.ascii
          - type: proton
            energy:    100.0  # MeV
            thickness: 60.0   # um
            filename:  pyxel/models/charge_generation/cosmix/data/stepsize_proton_100MeV_60um_Si_10k.ascii
          - type: proton
            energy:    100.0  # MeV
            thickness: 70.0   # um
            filename:  pyxel/models/charge_generation/cosmix/data/stepsize_proton_100MeV_70um_Si_10k.ascii
          - type: proton
            energy:    100.0  # MeV
            thickness: 100.0   # um
            filename:  pyxel/models/charge_generation/cosmix/data/stepsize_proton_100MeV_100um_Si_10k.ascii



.. autofunction:: pyxel.models.charge_generation.cosmix

.. _Dark current:

Dark current
============

:guilabel:`Charge` → :guilabel:`Charge`

With this model you can add a temperature dependent dark current to charge data,
stored in the a :py:class:`~pyxel.detectors.Detector` object.
The model follows the description in :cite:p:`Konnik:noises`.
The average dark current rate (in :math:`\mathit{e^-/s/pixel}`) is:

:math:`D_R = \frac{D_{FM}P_{S}}{q}\frac{T^\frac{3}{2}e^{-\frac{E_{gap}}{2k_{B}T}}}{T_{room}^\frac{3}{2}e^{-\frac{E_{g,room}}{2k_{B}T_{room}}}}`

where

:math:`T` is temperature (in :math:`K`), :math:`T_{room}` room temperature (:math:`\mathit{300 K}`), :math:`E_{g}` band gap (in :math:`eV`),
:math:`k_B` Boltzmann constant, :math:`D_{FM}` dark current figure of merit (in :math:`nA/cm^{2}`),
:math:`P_S` pixel area (in :math:`cm^{2}`), :math:`q` charge of an electron (in :math:`C`)and :math:`E_{g, room}` band gap at room temperature.
The entire dark current during exposure is:

:math:`I_{dark}=\mathcal{P}\big(t_{exp}D_R\big)\bigg(1+\mathcal{lnN}\big(0, \sigma^2_{fpn}\big)\bigg)`,

where :math:`\sigma_{fpn}=t_{exp} D_R D_N`, :math:`\mathcal{P}` Poisson distribution,
:math:`\mathcal{lnN}` log-normal distribution, :math:`D_N` the dark current spatial noise factor
and :math:`t_{exp}` exposure time (in :math:`s`).

To use the model,
user has to provide arguments ``figure_of_merit`` in :math:`\mathit{nA/cm^2}` (:math:`D_{FM}`),
``band_gap`` in :math:`\mathit{eV}`, ``band_gap_room_temperature`` in :math:`\mathit{eV}`, ``spatial_noise_factor`` (:math:`D_N`)
and ``temporal_noise``.
If ``temporal_noise`` is true, shot noise will be included.
The ``spatial_noise_factor`` is typically between 0.1 and 0.4 for CCD and CMOS sensors :cite:p:`Konnik:noises`.

Parameter ``temperature`` in :math:`\mathit{K}` is taken from detector :py:class:`~pyxel.detectors.Environment`.
If arguments ``band_gap`` and ``band_gap_room_temperature`` are not provided,
the model will use the Varshni empirical formula (see :cite:p:`VARSHNI1967149`) with parameters for Silicon by default:

:math:`E_{gap}(T) = E_{gap}(0) - \frac{\alpha T^2}{T+\beta}`.

For Silicon, material constants are :math:`E_{gap}(0)=1.1577\mathit{[eV]}`, :math:`\alpha=7.021\times10^{-4}\mathit{[eV/K]}`,
and :math:`\beta=1108\mathit{[K]}`.

Example of the configuration file:

.. code-block:: yaml

    - name: dark_current
      func: pyxel.models.charge_generation.dark_current
      enabled: true
      arguments:
        figure_of_merit: 1.  # nA/cm^2
        band_gap: 1.2  # eV, optional
        band_gap_room_temperature: 1.2  # eV, optional
        spatial_noise_factor: 0.1
        temporal_noise: false

.. note::
    You can find an example of this model used in this Jupyter Notebook
    :external+pyxel_data:doc:`examples/models/dark_current/dark_current_Si`
    from `Pyxel Data <https://esa.gitlab.io/pyxel-data>`_.

.. autofunction:: pyxel.models.charge_generation.dark_current

.. _Dark current rule07:

Dark current rule07
===================

.. note:: This model is specific for the :term:`MCT` detector.

:guilabel:`Charge` → :guilabel:`Charge`

With this model you can add dark current to :py:class:`~pyxel.data_structure.Charge` following the
model described in :cite:p:`Tennant2008MBEHT`.
This model is only valid for :term:`MCT` hybridised array (:term:`MCT`).
If ``temporal_noise`` is true, shot noise will be included.
The model has one extra argument: ``cut-off wavelength``, and also takes some values from :py:class:`~pyxel.detectors.Detector` object,
to be precise: ``temperature``, ``pixel size`` (assuming it is square),
and ``time step`` since last read-out.
Please make sure the detector :py:class:`~pyxel.detectors.Environment`, :py:class:`~pyxel.detectors.Geometry` and
:py:class:`~pyxel.detectors.Characteristics` are properly set in the ``YAML`` configuration file.

Example of the configuration file:

.. code-block:: yaml

    - name: dark_current
      func: pyxel.models.charge_generation.dark_current_rule07
      enabled: true
      arguments:
        cutoff_wavelength: 2.5
        spatial_noise_factor: 0.1
        temporal_noise: true

.. note::
    You can find an example of this model used in this Jupyter Notebook
    :external+pyxel_data:doc:`examples/models/dark_current/dark_current_rule07`
    from `Pyxel Data <https://esa.gitlab.io/pyxel-data>`_.

.. autofunction:: pyxel.models.charge_generation.dark_current_rule07    

.. _Simple dark current:

Simple dark current
===================

:guilabel:`Charge` → :guilabel:`Charge`

With this model you can add dark current to a :py:class:`~pyxel.detectors.Detector` object.

Example of the configuration file:

.. code-block:: yaml

    - name: simple_dark_current
      func: pyxel.models.charge_generation.simple_dark_current
      enabled: true
      arguments:
        dark_rate: 10.0

.. autofunction:: pyxel.models.charge_generation.simple_dark_current

.. _APD gain:

APD gain
========

.. note:: This model is specific to the :term:`APD` detector.

:guilabel:`Charge` → :guilabel:`Charge`

With this model you can apply APD gain to the a :py:class:`~pyxel.detectors.APD` object.
Model simply multiplies the values of charge with the avalanche gain,
which should be specified in the detector characteristics.

Example of the configuration file:

.. code-block:: yaml

    - name: apd_gain
      func: pyxel.models.charge_generation.apd_gain
      enabled: true

.. note::
    You can find an example of this model used in this Jupyter Notebook
    :external+pyxel_data:doc:`use_cases/APD/saphira`
    from `Pyxel Data <https://esa.gitlab.io/pyxel-data>`_.

.. autofunction:: pyxel.models.charge_generation.apd_gain

.. _Dark current Saphira:

Dark current Saphira
====================

.. note:: This model is specific to the :term:`APD` detector.

:guilabel:`Charge` → :guilabel:`Charge`

With this empirical model you can add dark current to a :py:class:`~pyxel.detectors.APD` object.
The model is an approximation the dark current vs. gain vs. temp plot in :cite:p:`2019:baker`, Fig. 3.
We can split it into three linear 'regimes': 1) low-gain, low dark current; 2) nominal; and 3) trap-assisted tunneling.
The model ignores the first one for now since this only applies at gains less than ~2.
All the necessary arguments are provided through the detector characteristics.
The model works best for ``temperature`` less than 100 and ``avalanche gain`` more than 2.

Example of the configuration file:

.. code-block:: yaml

    - name: dark_current_saphira
      func: pyxel.models.charge_generation.dark_current_saphira
      enabled: true

.. note:: Dark current calculated with this model already takes into account the avalanche gain.

.. note::
    You can find an example of this model used in this Jupyter Notebook
    :external+pyxel_data:doc:`use_cases/APD/saphira`
    from `Pyxel Data <https://esa.gitlab.io/pyxel-data>`_.

.. autofunction:: pyxel.models.charge_generation.dark_current_saphira


.. _Dark current induced:

Radiation induced Dark Current
==============================

:guilabel:`Charge` → :guilabel:`Charge`

This model adds dark current induced by radiation. A more detailed description of the models can be found in
:cite:p:`RadiationLeRoch2019` and :cite:p:`Belloir:16`.

Example of configuration file:

.. code-block:: yaml

    - name: radiation_induced_dark_current
      func: pyxel.models.charge_generation.radiation_induced_dark_current
      enabled: true
      arguments:
        depletion_volume: 64 # µm3
        annealing_time: 0.1 # weeks
        displacement_dose:  50  # TeV/g
        shot_noise: false

.. note::
    You can find an example of this model used in this Jupyter Notebook
    :external+pyxel_data:doc:`examples/models/dark_current_induced/dark_current_induced`
    from `Pyxel Data <https://esa.gitlab.io/pyxel-data>`_.

.. autofunction:: pyxel.models.charge_generation.radiation_induced_dark_current

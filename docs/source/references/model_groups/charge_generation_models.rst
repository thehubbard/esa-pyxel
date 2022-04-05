.. _charge_generation:

========================
Charge Generation models
========================

.. currentmodule:: pyxel.models.charge_generation
.. automodule:: pyxel.models.charge_generation

.. _Simple photoconversion:

Simple photoconversion
======================

:guilabel:`Photon` 🠆 :guilabel:`Charge`

With this model you can create and add charge to :py:class:`~pyxel.detectors.Detector` via photoelectric effect
by converting photons in charge. User can provide an optional quantum efficiency (``qe``) parameter.
If not provided, quantum efficiency from detector :py:class:`~pyxel.detectors.Characteristics` is used.

Basic example of YAML configuration model:

.. code-block:: yaml

    - name: simple_conversion
      func: pyxel.models.charge_generation.simple_conversion
      enabled: true
      arguments:
        qe: 0.8  # optional

.. autofunction:: simple_conversion

.. _Conversion with custom QE map:

Conversion with custom QE map
=============================

:guilabel:`Photon` 🠆 :guilabel:`Charge`

With this model you can create and add charge to :py:class:`~pyxel.detectors.Detector` via photoelectric effect
by converting photons in charge.
Beside that, user can input a custom quantum efficiency map by providing a ``filename`` of the :term:`QE` map.
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

.. autofunction:: conversion_with_qe_map

.. _Load charge:

Load charge
===========

:guilabel:`Charge` 🠆 :guilabel:`Charge`

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
      func: pyxel.models.photon_generation.load_charge
      enabled: true
      arguments:
        charge_file: data/charge.npy
        position: [0,0]

.. autofunction:: load_charge

.. _Charge injection:

Charge injection
================

:guilabel:`Charge` 🠆 :guilabel:`Charge`

With this model you can inject arbitrary charge block into rows of a :py:class:`~pyxel.detectors.CCD` detector.
Charge will be injected uniformly from row number `block_start` to row number `block_end`.

Example of YAML configuration model:

.. code-block:: yaml

    - name: charge_blocks
      func: pyxel.models.charge_generation.charge_blocks
      enabled: true
      arguments:
        charge_level: 100
        block_start: 10
        block_end: 50

.. note:: This model is specific for the :term:`CCD` detector.

.. autofunction:: charge_blocks

.. _CosmiX cosmic ray model:

CosmiX cosmic ray model
=======================

:guilabel:`Charge` 🠆 :guilabel:`Charge`

A cosmic ray event simulator was the first model added to Pyxel.
Initially it was a simple, semi-analytical model in Fortran using the stopping
power curve of protons to optimize the on-board source detection algorithm
of the Gaia telescope to discriminate between stars and cosmic rays. Then it
was reimplemented in Python as TARS (Tools for Astronomical Radiation
Simulations) and later as CosmiX.

With this model you can add the effect of cosmic rays to the :py:class:`~pyxel.data_structure.Charge` data structure.
See the documentation below for descriptions of parameters.
CosmiX model is described in detail in :cite:p:`2020:cosmix`.

Example of the configuration file:

.. code-block:: yaml

    - name: cosmix
      func: pyxel.models.charge_generation.cosmix
      enabled: true
      arguments:
        simulation_mode: cosmic_ray
        running_mode: "stepsize"
        particle_type: proton
        initial_energy: 100.          # MeV
        particle_number: 100
        incident_angles:
        starting_position:
        spectrum_file: 'data/proton_L2_solarMax_11mm_Shielding.txt'
        seed: 4321

.. autofunction:: pyxel.models.charge_generation.cosmix

.. _Dark current rule07:

Dark current rule07
===================

:guilabel:`Charge` 🠆 :guilabel:`Charge`

With this model you can add dark current to :py:class:`~pyxel.data_structure.Charge` following the
model described in :cite:p:`Tennant2008MBEHT`.
This model is only valid for :term:`MCT` hybridised array (:term:`MCT` + :term:`CMOS`).
The model has one extra argument: ``cut-off wavelength``,and also takes some values from :py:class:`~pyxel.detectors.Detector` object,
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

.. note:: This model is specific for the :term:`MCT` and :term:`CMOS` detector.

.. autofunction:: pyxel.models.charge_generation.dark_current_rule07    

.. _Dark current:

Dark current
============

:guilabel:`Charge` 🠆 :guilabel:`Charge`

With this model you can add dark current to a :py:class:`~pyxel.detectors.Detector` object.

Example of the configuration file:

.. code-block:: yaml

    - name: dark_current
      func: pyxel.models.charge_generation.dark_current
      enabled: true
      arguments:
        dark_rate: 10.0

.. autofunction:: pyxel.models.charge_generation.dark_current

.. _APD gain:

APD gain
========

:guilabel:`Charge` 🠆 :guilabel:`Charge`

With this model you can apply APD gain to the a :py:class:`~pyxel.detectors.APD` object.
Model simply multiplies the values of charge with the avalanche gain,
which should be specified in the detector characteristics.

Example of the configuration file:

.. code-block:: yaml

    - name: apd_gain
      func: pyxel.models.charge_generation.apd_gain
      enabled: true

.. note:: This model is specific to the :term:`APD` detector.

.. autofunction:: pyxel.models.charge_generation.apd_gain

.. _Dark current Saphira:

Dark current Saphira
====================

:guilabel:`Charge` 🠆 :guilabel:`Charge`

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

.. note:: This model is specific to the :term:`APD` detector.

.. note:: Dark current calculated with this model already takes into account the avalanche gain.

.. autofunction:: pyxel.models.charge_generation.dark_current_saphira

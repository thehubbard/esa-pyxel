.. _charge_generation:

========================
Charge Generation models
========================

.. currentmodule:: pyxel.models.charge_generation
.. automodule:: pyxel.models.charge_generation


Simple photoconversion
======================

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

Conversion with custom QE map
=============================

With this model you can create and add charge to :py:class:`~pyxel.detectors.Detector` via photoelectric effect
by converting photons in charge.
Beside that, user can input a custom quantum efficiency map by providing a ``filename`` of the QE map.
Accepted file formats for QE map are ``.npy``, ``.fits``, ``.txt``, ``.data``, ``.jpg``, ``.jpeg``, ``.bmp``,
``.png`` and ``.tiff``. Use argument ``position`` to set the offset from (0,0) pixel
and set where the input QE map is placed onto detector. You can set preset positions with argument ``align``.
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

Load charge
===========

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

Charge injection
================

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

.. note:: This model is specific for the CCD detector.

.. autofunction:: charge_blocks


TARS cosmic ray model
=====================

A cosmic ray event simulator was the first model added to Pyxel.
Initially it was a simple, semi-analytical model in Fortran using the stopping
power curve of protons to optimize the on-board source detection algorithm
of the Gaia telescope to discriminate between stars and cosmic rays. Then it
was reimplemented in Python as TARS (Tools for Astronomical Radiation
Simulations).

It is now being upgraded to either randomly sample distributions pre-generated
using Geant4 Monte Carlo particle transport simulations or directly call a
Geant4 application for each single event. The validation of the latest
version of the model against cosmic ray signals of the Gaia Basic Angle
Monitor CCDs is ongoing via Pyxel.

.. autofunction:: pyxel.models.charge_generation.run_tars


Dark current rule07
===================

With this model you can add dark current to :py:class:`~pyxel.data_structure.Charge` following the
model described in :cite:p:`Tennant2008MBEHT`.
This model is only valid for MCT hybridised array (MCT + CMOS).
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

.. note:: This model is specific for the MCT and CMOS detector.

.. autofunction:: pyxel.models.charge_generation.dark_current_rule07    


Dark current
============

With this model you can add dark current to a :py:class:`~pyxel.detectors.Detector` object.

Example of the configuration file:

.. code-block:: yaml

    - name: dark_current
      func: pyxel.models.charge_generation.dark_current
      enabled: true
      arguments:
        dark_rate: 10.0

.. autofunction:: pyxel.models.charge_generation.dark_current

.. _charge_generation:

Charge Generation models
========================

.. currentmodule:: pyxel.models.charge_generation
.. automodule:: pyxel.models.charge_generation


Simple photoconversion
----------------------

.. autofunction:: simple_conversion

Custom quantum efficiency map
-----------------------------

.. autofunction:: qe_map


TARS cosmic ray model
---------------------

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

..
    CCD charge injection
    --------------------

    .. autofunction:: pyxel.models.charge_generation.charge_injection.charge_injection

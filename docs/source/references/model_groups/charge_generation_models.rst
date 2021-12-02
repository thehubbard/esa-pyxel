.. _charge_generation:

Charge Generation models
========================

.. currentmodule:: pyxel.models.charge_generation
.. automodule:: pyxel.models.charge_generation


Simple photoconversion
----------------------

.. autofunction:: simple_conversion

Charge injection
----------------

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

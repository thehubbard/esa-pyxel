.. _charge_transfer:

============================
Charge Transfer models (CCD)
============================

.. important::
    This model group is only for CCD detectors!

.. currentmodule:: pyxel.models.charge_transfer
.. automodule:: pyxel.models.charge_transfer


Charge Distortion Model (CDM)
=============================

The Charge Distortion Model (CDM) describes the effects of the radiation
damage causing charge deferral and image shape distortion. The analytical
model is physically realistic, yet fast enough. It was developed specifically
for the Gaia CCD operating mode, implemented in Fortran and Python. However,
a generalized version has already been applied in a broader context, for
example to investigate the impact of radiation damage on the Euclid mission.
This generalized version has been included and used in Pyxel.

.. autofunction:: cdm


Arctic Add
==========

Add trap species.


Example of the configuration file:

.. code-block:: yaml

    - name: optical_psf
      func: pyxel.models.optics.optical_psf
      enabled: true
      arguments:
        well_fill_power: 10.
        trap_densities: [1., 2., 3.]                # Add three traps
        trap_release_timescales: [10., 20., 30.]
        express: 0


.. autofunction:: arctic_add

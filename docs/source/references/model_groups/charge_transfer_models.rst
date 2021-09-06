.. _charge_transfer:

Charge Transfer models (CCD)
============================

.. important::
    This model group is only for CCD detectors!

.. currentmodule:: pyxel.models.charge_transfer
.. automodule:: pyxel.models.charge_transfer


Charge Distortion Model (CDM)
-----------------------------

The Charge Distortion Model (CDM) describes the effects of the radiation
damage causing charge deferral and image shape distortion. The analytical
model is physically realistic, yet fast enough. It was developed specifically
for the Gaia CCD operating mode, implemented in Fortran and Python. However,
a generalized version has already been applied in a broader context, for
example to investigate the impact of radiation damage on the Euclid mission.
This generalized version has been included and used in Pyxel.

.. autofunction:: cdm

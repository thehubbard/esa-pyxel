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


Add CTI trails
==============

Add image trails due to charge transfer inefficiency in CCD detectors by modelling the
trapping, releasing, and moving of charge along pixels.

The primary inputs are the initial image followed by the properties of the CCD,
readout electronics and trap species for serial clocking.

More information about adding CTI trailing is described
in section 2.1 in :cite:p:`2010:massey`.


Example of the configuration file:

.. code-block:: yaml

    - name: arctic_add
      func: pyxel.models.charge_transfer.arctic_add
      enabled: true
      arguments:
        well_fill_power: 10.
        trap_densities: [1., 2., 3.]                # Add three traps
        trap_release_timescales: [10., 20., 30.]
        express: 0


.. autofunction:: arctic_add


Remove CTI trails
=================

Remove CTI trails is done by iteratively modelling the addition of CTI, as described
in :cite:p:`2010:massey` section 3.2 and Table 1.

Example of the configuration file:

.. code-block:: yaml

    - name: arctic_remove
      func: pyxel.models.charge_transfer.arctic_remove
      enabled: true
      arguments:
        well_fill_power: 10.
        instant_traps:                      # Add two traps
          - density: 1.0
            release_timescale: 10.0
          - density: 2.0
            release_timescale: 20.0
        express: 0


.. autofunction:: arctic_remove

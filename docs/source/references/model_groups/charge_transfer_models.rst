.. _charge_transfer:

============================
Charge Transfer models (CCD)
============================

.. important::
    This model group is only for CCD detectors!

.. currentmodule:: pyxel.models.charge_transfer


Charge Distortion Model (CDM)
=============================

The Charge Distortion Model - CDM :cite:p:`2013:short` describes the effects of the radiation
damage causing charge deferral and image shape distortion. The analytical
model is physically realistic, yet fast enough. It was developed specifically
for the Gaia CCD operating mode, implemented in Fortran and Python. However,
a generalized version has already been applied in a broader context, for
example to investigate the impact of radiation damage on the Euclid mission.
This generalized version has been included and used in Pyxel.

Use this model to add radiation induced CTI effects to :py:class:`~pyxel.data_structure.Pixel` array of the
to :py:class:`~pyxel.detectors.CCD` detector. Argument ``direction`` should be set as either ``"parallel"``
for parallel direction CTI or ``"serial"`` for serial register CTI.
User should also set arguments ``trap_release_times``, ``trap_densities`` and ``sigma``
as lists for an arbitrary number of trap species. See below for descriptions.
Other arguments include ``max_electron_volume``, ``transfer_period``,
``charge injection`` for parallel mode and ``full_well_capacity`` to override the one set in
detector :py:class:`~pyxel.detectors.Characteristics`.

Example of the configuration file.

.. code-block:: yaml

    - name: cdm
      func: pyxel.models.charge_transfer.cdm
      enabled: true
      arguments:
        direction: "parallel"
        trap_release_times: [0.1, 1.]
        trap_densities: [0.307, 0.175]
        sigma: [1.e-15, 1.e-15]
        beta: 0.3
        max_electron_volume: 1.e-10,
        transfer_period: float = 1.e-4,
        charge_injection: true  # only used for parallel mode
        full_well_capacity: 1000.  # optional (otherwise one from detector characteristics is used)

.. note:: This model is specific for the CCD detector.

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

.. _optical:

==============
Optical models
==============

.. currentmodule:: pyxel.models.optics
.. automodule:: pyxel.models.optics


Physical Optics Propagation in PYthon (POPPY)
=============================================

.. automodule:: pyxel.models.optics.poppy

Example of the configuration file:

.. code-block:: yaml

    - name: optical_psf
      func: pyxel.models.optics.optical_psf
      enabled: true
      arguments:
        fov_arcsec: 5               # FOV in arcseconds
        pixelscale: 0.01            # arcsec/pixel
        wavelength: 0.6e-6          # wavelength in meters
        optical_system:
          - item: CircularAperture
            radius: 3.0


.. autofunction:: optical_psf


Simple optical alignment
========================

.. automodule:: pyxel.models.optics.alignment

Example of the configuration file:

.. code-block:: yaml

    - name: alignment
      func: pyxel.models.optics.alignment
      enabled: true

.. autofunction:: alignment

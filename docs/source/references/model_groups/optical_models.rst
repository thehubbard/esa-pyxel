.. _optical:

==============
Optical models
==============

.. currentmodule:: pyxel.models.optics
.. automodule:: pyxel.models.optics


Physical Optics Propagation in PYthon (POPPY)
=============================================

:guilabel:`Photon` 🠆 :guilabel:`Photon`

POPPY (Physical Optics Propagation in Python) model wrapper.

It calculates the optical Point Spread Function of an optical system and applies the convolution.

Documentation:
https://poppy-optics.readthedocs.io

See details about POPPY Optical Element classes:
https://poppy-optics.readthedocs.io/en/stable/available_optics.html

Supported optical elements:

- ``CircularAperture``
- ``SquareAperture``
- ``RectangularAperture``
- ``HexagonAperture``
- ``MultiHexagonalAperture``
- ``ThinLens``
- ``SecondaryObscuration``
- ``ZernikeWFE``
- ``SineWaveWFE``


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
        optical_system:
          - item: CircularAperture
            radius: 1.5
          - item: ThinLens
            radius: 1.2
            nwaves: 1
          - item: ZernikeWFE
            radius: 0.8
            coefficients: [0.1e-6, 3.e-6, -3.e-6, 1.e-6, -7.e-7, 0.4e-6, -2.e-6]
            aperture_stop: false

.. autofunction:: optical_psf

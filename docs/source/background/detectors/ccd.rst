.. _CCD architecure:

###
CCD
###

API reference: :py:class:`~pyxel.detectors.CCD`

Available models
================

* Photon generation
    * :ref:`Load image`
    * :ref:`Simple illumination`
    * :ref:`Stripe pattern`
    * :ref:`Shot noise`
* Optics
    * :ref:`Physical Optics Propagation in PYthon (POPPY)`
    * :ref:`Load PSF`
* Charge generation
    * :ref:`Simple photoconversion`
    * :ref:`Conversion with custom QE map`
    * :ref:`Load charge`
    * :ref:`Charge injection`
    * :ref:`CosmiX cosmic ray model`
    * :ref:`Dark current`
* Charge collection
    * :ref:`Simple collection`
    * :ref:`Simple full well`
    * :ref:`Fixed pattern noise`
* Charge transfer
    * :ref:`Charge Distortion Model (CDM)`
    * :ref:`Add CTI trails (ArCTIc)`
    * :ref:`Remove CTI trails (ArCTIc)`
* Charge measurement:
    * :ref:`DC offset`
    * :ref:`Simple charge measurement`
    * :ref:`Output node noise`
    * :ref:`Non-linearity (polynomial)`
* Readout electronics:
    * :ref:`Simple ADC`
    * :ref:`Simple amplification`
    * :ref:`SAR ADC`
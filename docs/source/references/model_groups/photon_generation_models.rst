.. _photon_generation:

Photon Generation models
========================


.. currentmodule:: pyxel.models.photon_generation
.. automodule:: pyxel.models.photon_generation


Loading image
-------------

.. autofunction:: load_image

Simple illumination
-------------------

.. autofunction:: illumination

Stripe pattern
--------------

.. autofunction:: stripe_pattern

Shot noise
----------
Use this model to add shot noise to the :py:class:`~pyxel.data_structure.Photon` array.
By default (no arguments provided), the model uses the Poisson distribution (``numpy.random.poisson``).
User can also set the argument  ``type`` to ``"normal"`` for normal distribution (``numpy.random.normal``).
As known, for large photon counts :math:`N` the Poisson distribution approaches the normal distribution
with standard deviation :math:`\sqrt{N}`, which is fixed in the model.
It is also possible to set the seed of the random generator with the argument ```seed``.

Example of the configuration file:

.. code-block:: yaml

  - name: shot_noise
    func: pyxel.models.photon_generation.shot_noise
    enabled: true
    arguments:
      type: "poisson"  # optional

.. autofunction:: shot_noise

.. _detectors:

#########
Detectors
#########

According to the YAML file, one CCD or CMOS Detector object is instantiated
for each thread, inheriting from a general (abstract) Detector class.

The created Detector object is the input of the Detection pipeline, which is
passed through all the including models represented by functions. We can
consider the Detector object as a bucket containing all information and data
related to the physical properties of the simulated detector (geometry,
material, environment, characteristics), incident photons, created
charge-carriers and the generated signals we are interested in at the
end of the simulation.

.. figure:: _static/pyxel_detector.png
    :scale: 25%
    :alt: detector
    :align: center

.. _data_structure:

Data Structure
================

Models in Pyxel should be able to add photons, charges,
charge packets, signal or image pixel values to the corresponding
data structure classes (Photon, Charge, Pixel, Signal or Image class).

These classes are storing the data values
either inside a Pandas DataFrame or in a NumPy array. Via DataFrame or
NumPy array handling functions, models can also modify properties of photons,
charges, etc., like position, kinetic energy,
number of electrons per charge packet, signal amplitude, etc.

DataFrame and Array classes and their methods to store and handle the
data in Pyxel:
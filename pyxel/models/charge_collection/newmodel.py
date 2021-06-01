#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#

"""
newmodel module for the PyXel simulation

This module is used in charge_collection

-------------------------------------------------------------------------------

+--------------+----------------------------------+---------------------------+
| Author       | Name                             | Creation                  |
+--------------+----------------------------------+---------------------------+
| You          | newmodel                        | Tue Jun  1 08:40:52 2021                   |
+--------------+----------------------------------+---------------------------+

+-----------------+-------------------------------------+---------------------+
| Contributor     | Name                                | Creation            |
+-----------------+-------------------------------------+---------------------+
| Name            | filename                            | 06/21/2019          |
+-----------------+-------------------------------------+---------------------+

This is a documentation template for the newmodel module.
This docstring can be used for automatic doc generation and explain more
in detail what the newmodel module does in PyXel.

This module can be found in pyxel/models/charge_collection.
Please modify the docstrings accordingly to provide the users a simple and
detailed explanation of your algorithm for this module.

Table examples
==============

===============  ==============================================================
Table entries    Table values
===============  ==============================================================
Entry 1          Value 1
Entry 2          Value 2
Entry 3          Value 3

Entry 4          Value 4

Entry 5          Value 5
===============  ==============================================================

+------------------------+------------+----------+----------+
| Header row, column 1   | Header 2   | Header 3 | Header 4 |
| (header rows optional) |            |          |          |
+========================+============+==========+==========+
| body row 1, column 1   | column 2   | column 3 | column 4 |
+------------------------+------------+----------+----------+
| body row 2             | Cells may span columns.          |
+------------------------+------------+---------------------+
| body row 3             | Cells may  | - Table cells       |
+------------------------+ span rows. | - contain           |
| body row 4             |            | - body elements.    |
+------------------------+------------+----------+----------+
| body row 5             | Cells may also be     |          |
|                        | empty: ``-->``        |          |
+------------------------+-----------------------+----------+

Code example
============

.. code-block:: python

    import sys

    print('Hello world...')


.. literalinclude:: pyxel/models/charge_collection/newmodel.py
    :language: python
    :linenos:
    :lines: 84-87

Model reference in the YAML config file
=======================================

.. code-block: yaml

    pipeline:

      # Small comment on what it does
      charge_collection:
        - name: newmodel
          func: pyxel.models.charge_collection.newmodel.model
          enabled: true
          arguments:
            arg1: data/fits/Pleiades_HST.fits
            arg2: true
            arg3: 42

Useful links
============

ReadTheDocs documentation
https://sphinx-rtd-theme.readthedocs.io/en/latest/index.html

.. todo::

   Write the documentation for newmodel

"""
import typing as t

# One or the other
from pyxel.detectors import CMOS
from pyxel.detectors import CCD

def model(detector: CCD,
          arg1: str, arg2: bool = True, arg3: int = 42) -> None:
    """This method does nothing.

    Parameters
    ----------
    arg1: str
    arg2: bool
    arg3: int

    Returns
    -------
    None
    """

    #Access the detector
    """
    photon = detector.photon.array
    pixel = detector.pixel.array
    signal = detector.signal.array
    image = detector.image.array
    """
    # Do operation on one of those arrays and return None.

    # Done!

    return None

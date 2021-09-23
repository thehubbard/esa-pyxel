=============
Running Pyxel
=============

Pyxel can be run either from command line or used as a library, for example in Jupyter notebooks.

Running Pyxel from command line
===============================

To run Pyxel on your local computer, simply run it from the command-line:

.. code-block:: bash

    $ python pyxel/run.py -c input.yaml

    or

    $ pyxel --config input.yaml

    or

    $ python -m pyxel --config input.yaml


where

======  ===============  =======================================  ========
``-c``  ``--config``     defines the path of the input YAML file  required
``-s``  ``--seed``       defines a seed for random number         optional
                         generator
``-v``  ``--verbosity``  increases the output verbosity (-v/-vv)  optional
``-V``  ``--version``    prints the version of Pyxel              optional
======  ===============  =======================================  ========

Running Pyxel in jupyter notebooks
==================================

An example of running Pyxel as a library:

.. code-block:: python

    from pyxel.configuration import load
    from pyxel.run import single_mode

    configuration = load("configuration.yaml")
    single = configuration.single
    detector = configuration.ccd_detector
    pipeline = configuration.pipeline

    single_mode(single, detector, pipeline)

Running Pyxel from a Docker container
=====================================

If you want to run Pyxel in a Docker container, you must first get the source code
from the `Pyxel GitLab repository <https://gitlab.com/esa/pyxel>`_.

.. code-block:: console

    $ git clone https://gitlab.com/esa/pyxel.git
    $ cd pyxel


.. Note::
    Folder ‘pyxel/volumes/notebooks’ is linked to
    folder ‘/home/pyxel/notebooks’ in the container.


Create the container
--------------------

.. tab:: docker-compose

    .. code-block:: console

        $ docker-compose build

.. tab:: only docker

    .. code-block:: console

        $ docker build -t pyxel .


Start the container
-------------------

Run Pyxel with a Jupyter Lab server from a docker container:

.. tab:: docker-compose

    .. code-block:: console

        $ docker-compose up -d

.. tab:: only docker

    .. code-block:: console

        $ docker run -d -p 8888:8888 -v ./volumes/notebooks:/home/pyxel/notebooks pyxel


Stop the container
------------------

Stop a running Pyxel container.

.. tab:: docker-compose

    .. code-block:: console

        $ docker-compose down

.. tab:: only docker

    .. code-block:: console

        $ docker stop

Check if the container is running
----------------------------------

.. tab:: docker-compose

    .. code-block:: console

        docker-compose ps
            Name                   Command               State           Ports
        -------------------------------------------------------------------------------
        pyxel_pyxel_1   /bin/bash --login -c conda ...   Up      0.0.0.0:8888->8888/tcp


.. tab:: only docker

    .. code-block:: console

        docker ps


Get logs
--------

.. tab:: docker-compose

    .. code-block:: console

        docker-compose logs -f


.. tab:: only docker

    .. code-block:: console

        TBW

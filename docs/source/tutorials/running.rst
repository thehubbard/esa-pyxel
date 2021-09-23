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
    Folder ‘./pyxel/volumes/notebooks’ is linked to
    folder ‘/home/pyxel/jupyter/notebooks’ in the container.


Build an image
--------------

.. tab:: docker-compose

    .. code-block:: console

        Create docker image 'pyxel_pyxel'
        $ docker-compose build

.. tab:: only docker

    .. code-block:: console

        Create docker image 'pyxel'
        $ docker build -t pyxel .


Create and start the container
------------------------------

Run Pyxel with a Jupyter Lab server from a new docker container:

.. tab:: docker-compose

    .. code-block:: console

        Create and start a new container 'pyxel_pyxel_1'
        $ docker-compose up -d

.. tab:: only docker

    .. code-block:: console

        Create and start new container 'my_pyxel' from image 'pyxel'
        $ docker create -p 8888:8888 -v $PWD/volumes/notebooks:/home/pyxel/jupyter/notebooks pyxel --name my_pyxel
        $ docker start my_pyxel

Stop and remove the container
-----------------------------

Stop and remove a running Pyxel container.

.. tab:: docker-compose

    .. code-block:: console

        Stop and remove container 'pyxel_pyxel_1'
        $ docker-compose down

.. tab:: only docker

    .. code-block:: console

        Stop and remove container 'my_pyxel'
        $ docker stop my_pyxel
        $ docker rm my_pyxel

Check if the container is running
----------------------------------

List running containers.

.. tab:: docker-compose

    .. code-block:: console

        $ docker-compose ps


.. tab:: only docker

    .. code-block:: console

        $ docker ps


Get logs
--------

View output from the Pyxel container.

.. tab:: docker-compose

    .. code-block:: console

        Get logs from container 'pyxel_pyxel_1'
        $ docker-compose logs -f


.. tab:: only docker

    .. code-block:: console

        Get logs from container 'my_pyxel'
        $ docker logs -f my_pyxel

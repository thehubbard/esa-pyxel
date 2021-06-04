.. _install:

============
Installation
============

Pyxel can be installed with `conda <https://docs.conda.io/>`_,
`pip <https://pip.pypa.io/>`_ or from source.

If you want to have a full installation of Pyxel, then the recommended installation
method is to use `conda <https://docs.conda.io/>`__ into a conda environment.


Dependencies
============

Pyxel has the following **mandatory** dependencies:

* `python <https://www.python.org>`_ 3.7 or later
* `numpy <https://numpy.org>`_
* `xarray <http://xarray.pydata.org/>`_
* `dask <https://dask.org>`_
* `jupyterlab <https://jupyterlab.readthedocs.io>`_
* `astropy <https://www.astropy.org>`_
* `pandas <https://pandas.pydata.org>`_
* `numba <https://numba.pydata.org>`_
* `tqdm <https://tqdm.github.io>`_
* `holoviews <https://holoviews.org>`_
* `matplotlib <https://matplotlib.org>`_
* `h5py <https://www.h5py.org>`_

Additionally, Pyxel has the following **optional** dependencies:

* `pygmo <https://esa.github.io/pygmo2/>`_, version 2.10 or later
* `poppy <https://poppy-optics.readthedocs.io/>`_, version 0.8 or later

.. warning::
    It is **strongly** encouraged to install optional package
    `pygmo <https://esa.github.io/pygmo2/>`_ with `conda` rather than `pip`.
    See `here <https://esa.github.io/pygmo2/install.html#pip>`_ for more information.

    Moreover, only the binaries of `pygmo` for Linux (not MacOS or Windows)
    are available on `pip`.

    The binaries of `pygmo` for MacOS, Windows and Linux are only available
    on Conda 64bit (**not 32bit**).


.. note::
    Optional package `poppy <https://poppy-optics.readthedocs.io/>`_ is not available
    on `conda`, only on the `PyPI` repository.


Instructions
============

Pyxel itself is a pure Python package, but its dependencies are not.
The easiest way to get everything installed is to use
`conda <https://docs.conda.io>`__.

The following instructions are valid for MacOS, Windows and Linux.

.. note::
    Pyxel is not yet officially available on the `PyPI <https://pypi.org>`_ repository or
    any conda channels
    (e.g. `Conda-forge community channel <https://anaconda.org/conda-forge/repo>`_).


Conda
-----

If you use `conda`, you can install Pyxel with:

.. code-block:: bash

    # Best practice, use an environment rather than install in the base env
    conda create -n my-env
    conda activate my-env

    # Install Python and pip in the environment
    conda install python>=3.7 pip

    # Install Pygmo (needed for calibration)
    conda install -c conda-forge pygmo

    # Full installation of Pyxel including all dependencies
    pip install -e git+https://gitlab.com/esa/pyxel.git#egg=pyxel-sim[all]

.. warning::
    Conda 64-bit **must** be installed and not Conda 32-bit.


Pip
---

If you use pip, you can install Pyxel with:

.. code-block:: bash

    # Install Pyxel without optional dependencies
    pip install -e git+https://gitlab.com/esa/pyxel.git#egg=pyxel-sim

    # Or install Pyxel with some optional dependencies
    pip install -e git+https://gitlab.com/esa/pyxel.git#egg=pyxel-sim[calibration]
    pip install -e git+https://gitlab.com/esa/pyxel.git#egg=pyxel-sim[model]

    # Or install Pyxel with all optional dependencies
    pip install -e git+https://gitlab.com/esa/pyxel.git#egg=pyxel-sim[all]


To update Pyxel with pip, you can use the same previous instructions to install Pyxel.

When using pip, it's good practice to use a virtual environment.
See `this guide <https://dev.to/bowmanjd/python-tools-for-managing-virtual-environments-3bko#howto>`_
for details on using virtual environments.


Install from source
-------------------

To install Pyxel from source, clone the repository from the
`Pyxel GitLab repository <https://gitlab.com/esa/pyxel>`_

.. code-block:: bash

    # Get source code
    git clone https://gitlab.com/esa/pyxel.git
    cd pyxel
    python install -m pip install .

.. note::
    The `Pyxel GitLab repository <https://gitlab.com/esa/pyxel>`__ is not yet open to
    all users. To get access to this repository, you must contact the maintainers
    (pyxel at esa dot int).

You can install all dependencies as well:

.. code-block:: bash

    python -m pip install ".[all]"


Verify the installation
=======================

You can verify that Pyxel is installed with the following command:

.. code-block:: bash

    python -c "import pyxel; pyxel.show_versions()"


..
    Python
    ~~~~~~

    Before you got any further, make sure you've got Python 3.7 or newer available
    from your command line.

    You can check this by simply running:

    .. code-block:: bash

      $ python3 --version
      Python 3.7.2

      or

      $ python3.7 --version
      Python 3.7.2


    On Windows, you can also try:

    .. code-block:: bash

     $ py -3 --version
     Python 3.7.2

     or

     $ py -3.7 --version
     Python 3.7.2

    .. note::

      Do not use command ``python``, you should use a command like ``pythonX.Y``.
      For example, to start Python 3.7, you use the command ``python3.7``.


..
    Pip
    ~~~

    Furthermore, you'll need to make sure pip is installed with a recent version.
    You can check this by running:

    .. code-block:: bash

      $ python3.7 -m pip --version
      pip 19.1.1

    .. note::

      Do not use command ``pip`` but ``python -m pip``.
      For example, to start ``pip`` for Python 3.7, you use the
      command ``python3.7 -m pip``.

    You can find more information about installing packages
    at this `link <https://packaging.python.org/installing/>`_.


..
    Install from source
    ===================

    Get source code
    ~~~~~~~~~~~~~~~

    First, get access to the `Pyxel GitLab repository <https://gitlab.com/esa/pyxel>`_
    from maintainers (pyxel at esa dot int).

    If you can access it, then clone the GitLab repository to your computer
    using ``git``:

    .. code-block:: bash

        $ git clone https://gitlab.com/esa/pyxel.git


..
    Install requirements
    ~~~~~~~~~~~~~~~~~~~~

    After cloning the repository, install the dependency provided together
    with Pyxel using ``pip``:


    .. code-block:: bash

      $ cd pyxel
      $ python3.7 -m pip install -r requirements.txt

    .. note::
      This command installs all packages that cannot be found in ``pypi.org``.
      This step will disappear for future versions of ``pyxel``.

    .. important::
      To prevent breaking any system-wide packages (ie packages installed for all users)
      or to avoid using command ``$ sudo pip ...`` you can
      do a `user installation <https://pip.pypa.io/en/stable/user_guide/#user-installs>`_.

      With the command: ``$ python3.7 -m pip install --user -r requirements.txt``

..
    Install Pyxel
    ~~~~~~~~~~~~~

    To install ``pyxel`` use ``pip`` locally, choose one from
    the 4 different options below:


    .. code-block:: bash

      $ python3.7 -m pip install -e ".[all]"            # Install everything (recommended)
      $ python3.7 -m pip install -e ".[calibration]"    # Install dependencies for 'calibration mode' (pygmo)
      $ python3.7 -m pip install -e ".[model]"          # Install dependencies for optional models (poppy)
      $ python3.7 -m pip install -e .                   # Install without any optional dependencies


    ..
      To install ``pyxel`` use ``pip`` locally, choose one from the 4 different options below:

        * To install ``pyxel`` and all the optional dependencies (recommended):

        .. code-block:: bash

          $ python3.7 -m pip install -e ".[all]"

        * To install ``pyxel`` and the optional dependencies for *calibration mode* (``pygmo``):

        .. code-block:: bash

          $ python3.7 -m pip install -e ".[calibration]"

        * To install ``pyxel`` and the optional models (``poppy``):

        .. code-block:: bash

          $ python3.7 -m pip install -e ".[model]"

        * To install ``pyxel`` without any optional dependency:

        .. code-block:: bash

          $ python3.7 -m pip install -e .


    .. important::
      To prevent breaking any system-wide packages (ie packages installed for all users)
      or to avoid using command ``$ sudo pip ...`` you can do a `user installation <https://pip.pypa.io/en/stable/user_guide/#user-installs>`_.
      Whenvever you see the command ``$ python3.7 -m pip install ...`` then replace it
      by the command ``$ python3.7 -m pip install --user ...``.

      If ``pyxel`` is not available in your shell after installation, you will need to add
      the `user base <https://docs.python.org/3/library/site.html#site.USER_BASE>`_'s binary
      directory to your PATH.

      On Linux and MacOS the user base binary directory is typically ``~/.local``.
      You'll need to add ``~/.local/bin`` to your PATH.
      On Windows the user base binary directory is typically
      ``C:\Users\Username\AppData\Roaming\Python36\site-packages``.
      You will need to set your PATH to include
      ``C:\Users\Username\AppData\Roaming\Python36\Scripts``.
      you can find the user base directory by running
      ``python3.7 -m site --user-base`` and adding ``bin`` to the end.


    After the installation steps above,
    see :ref:`here how to run Pyxel <running_modes>`.

..
    Install from PyPi
    -----------------

    TBW.


    To upgrade ``pyxel`` to the latest version:

    TBW.

..
    Install with Anaconda
    ---------------------

    TBW.

    .. note::
      If a package is not available in any PyPI server for your OS, because
      you are using Conda or Anaconda Python distribution, then you might
      have to download the Conda compatible whl file of some dependencies
      and install it manually with ``conda install``.

      If you use OSX, then you can only install ``pygmo`` with Conda.

..
    Using Docker
    -------------

    TBW.

..
    Installation with Anaconda
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    First install the `Anaconda distribution <https://www.anaconda.com/distribution/>`_
    then check if the tool ``conda`` is correctly installed:

    .. code-block:: bash

      $ conda info

    The second step is to create a new conda environment `pyxel-dev` and
    to install the dependencies with ``conda`` and ``pip``:

    .. code-block:: bash

      $ cd pyxel

      Create a new conda environment 'pyxel-dev'
      and install some dependencies from conda with `environment.yml`
      $ conda env create -f environment.yml

      Display all conda environments (only for checking)
      $ conda info --envs

      Activate the conda environment 'pyxel-dev'
      $ (pyxel-dev) conda activate pyxel-dev

      Install the other dependencies not installed by conda
      $ (pyxel-dev) pip install -r requirements.txt


    Then install ``pyxel`` in the conda environment:

    .. code-block:: bash

      $ (pyxel-dev) cd pyxel
      $ (pyxel-dev) pip install -e .

    More about the conda environments (only for information):

    .. code-block:: bash

      Deactivate the environment
      $ conda deactivate

      Remove the conda environment 'pyxel-dev'
      $ conda remove --name pyxel-dev --all

    After the installation steps above,
    see :ref:`here how to run Pyxel <running_modes>`.


    Using Docker
    -------------

    .. attention::
        Not yet available!

    Using Docker, you can just download the Pyxel Docker image and run it without
    installing Pyxel.

    How to run a Pyxel container with Docker:

    Login:

    .. code-block:: bash

      docker login gitlab.esa.int:4567

    Pull latest version of the Pyxel Docker image:

    .. code-block:: bash

      docker pull gitlab.esa.int:4567/sci-fv/pyxel

    Run Pyxel Docker container with GUI:

    .. code-block:: bash

      docker run -p 9999:9999 \
                 -it gitlab.esa.int:4567/sci-fv/pyxel:latest \
                 --gui True

    Run Pyxel Docker container in batch mode (without GUI):

    .. code-block:: bash

      docker run -p 9999:9999 \
                 -v C:\dev\work\docker:/data \
                 -it gitlab.esa.int:4567/sci-fv/pyxel:latest \
                 -c /data/settings_ccd.yaml \
                 -o /data/result.fits

    List your running Docker containers:

    .. code-block:: bash

      docker ps

    After running Pyxel container you can access it:

    .. code-block:: bash

      docker exec -it <CONTAINER_NAME> /bin/bash

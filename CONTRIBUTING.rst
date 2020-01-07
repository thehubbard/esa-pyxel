.. _contributing:

******************
Contribution guide
******************

.. note::

  Large parts of this document came from the `Pandas Contributing
  Guide <http://pandas.pydata.org/pandas-docs/stable/contributing.html>`_.


Pyxel is built with modularity and flexibility in mind and makes use of
the GitLab infrastructure for version control and to support collaborative
development. Therefore it shall be easy for its user & developer
community to directly contribute and expand the framework capabilities by
adding their own models, codes simulating physical processes and effects of
detectors.

Where to start?
===============

All contributions, bug reports, bug fixes, documentation improvements, 
and ideas are welcome.


If you are brand new to ``Pyxel`` or open-source development, we recommend going through
the `GitLab "issues" <https://gitlab.com/esa/pyxel/issues>`_ tab to find issues 
that interest you.
There a number of issues listed under `Documentation <https://gitlab.com/esa/pyxel/issues?label_name%5B%5D=documentation>`_
and `good first issue <https://gitlab.com/esa/pyxel/issues?label_name%5B%5D=good+first+issue>`_
where you could start out.
Once you've found an interesting issue, you can return here to get your development
environment setup.

Feel free to ask question on the `mailing list`_

.. _contributing.bug_reports:


Bug reports and enhancement requests
====================================

Bug reports are an important part of making ``Pyxel`` more stable.
Having a complete bug report will allow others to reproduce the bug and provide
insight into fixing.
See `this stackoverflow article <https://stackoverflow.com/help/mcve>`_ for tips on
writing a good bug report.

Trying the bug-producing code out on the *master* branch is often a worthwhile exercise
to confirm the bug still exists. It is also worth searching existing bug reports and
pull requests to see if the issue has already been reported and/or fixed.

Bug reports must:

#. Include a short, self-contained python snipper reproducing the problem.
   You can format the code nicely by using `GitLab Flavored Markdown
   <https://docs.gitlab.com/ee/user/markdown.html#gitlab-flavored-markdown-gfm>`_::

      ```python
      >>> from pyxel.io import load
      >>> cfg = load('config.yml')
      ...
      ```

#. Include the full version string of ``Pyxel`` and its dependencies. You can use the
   built in function::

   >>> import pyxel
   >>> pyxel.__version__

#. Explain why the current behavior is wrong/not desired and what you expect instead.

The issue will be show up to the ``Pyxel`` community and be open to comments/ideas
from others.

.. _contributing.gitlab:


Working with the code
=====================

Now that you have an issue you want to fix, enhancement to add, or documentation
to improve, you need to learn how to work with GitLab and the ``Pyxel`` code base.

.. _contributing.version_control:

Version control, Git, and GitLab
--------------------------------

Now that you have an issue you want to fix, enhancement to add, or documentation
to improve, you need to learn how to work with GitHub and the *xarray* code base.

.. _contributing.version_control:

Version control, Git, and GitHub
--------------------------------

To the new user, working with Git is one of the more daunting aspects of contributing
to ``Pyxel``.  It can very quickly become overwhelming, but sticking to the guidelines
below will help keep the process straightforward and mostly trouble free.  As always,
if you are having difficulties please feel free to ask for help.

The code is hosted on `GitLab <https://gitlab.com/esa/pyxel>`_. To
contribute you will need to sign up for a `free GitLab account
<https://gitlab.com/users/sign_in#register-pane>`_. We use `Git <http://git-scm.com/>`_ for
version control to allow many people to work together on the project.

Some great resources for learning Git:

* the `GitLab help pages <https://docs.gitlab.com>`_.
* the `NumPy's documentation <http://docs.scipy.org/doc/numpy/dev/index.html>`_.
* Matthew Brett's `Pydagogue <http://matthew-brett.github.com/pydagogue/>`_.


Getting started with Git
------------------------

`GitLab has instructions <https://docs.gitlab.com/ee/gitlab-basics/start-using-git.html>`__ for installing git,
setting up your SSH key, and configuring git.  All these steps need to be completed before
you can work seamlessly between your local repository and GitLab.

.. _contributing.forking:

Forking
-------

You will need your own fork to work on the code. Go to the `Pyxel project
page <https://gitlab.com/esa/pyxel>`_ and hit the ``Fork`` button. You will
want to clone your fork to your machine::

    git clone https://gitlab.com/your-user-name/pyxel.git
    cd pyxel
    git remote add upstream https://gitlab.com/esa/pyxel.git

This creates the directory `Pyxel` and connects your repository to
the upstream (main project) ``Pyxel`` repository.

.. _contributing.dev_env:

Creating a development environment
----------------------------------

To test out code changes, you'll need to build ``Pyxel`` from source, which
requires a Python environment. If you're making documentation changes, you can
skip to :ref:`contributing.documentation` but you won't be able to build the
documentation locally before pushing your changes.

.. _contributing.dev_python:


Creating a Python Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before starting any development, you'll need to create an isolated xarray
development environment:

- Install either `Anaconda <https://www.anaconda.com/download/>`_ or `miniconda
  <https://conda.io/miniconda.html>`_
- Make sure your conda is up to date (``conda update conda``)
- Make sure that you have :ref:`cloned the repository <contributing.forking>`
- ``cd`` to the ``Pyxel`` source directory

We'll now kick off a two-step process:

1. Install the build dependencies
2. Build and install Pyxel

.. code-block:: none

   # Create and activate the build environment
   conda env create -f requirements.yml
   conda activate pyxel-dev

   # or with older versions of Anaconda:
   source activate pyxel-dev

   # Build and install pyxel
   pip install -e .

At this point you should be able to import ``Pyxel```` from your locally built version::

   $ python  # start an interpreter
   >>> import pyxel
   >>> pyxel.__version__
   '0.5+0.gcae5a0b'

This will create the new environment, and not touch any of your existing environments,
nor any existing Python installation.

To view your environments::

      conda info -e

To return to your root environment::

      conda deactivate

See the full conda docs `here <http://conda.pydata.org/docs>`__.


Creating a branch
-----------------

You want your master branch to reflect only production-ready code, so create a
feature branch for making your changes. For example::

    git branch shiny-new-feature
    git checkout shiny-new-feature

The above can be simplified to::

    git checkout -b shiny-new-feature

This changes your working directory to the shiny-new-feature branch.  Keep any
changes in this branch specific to one bug or feature so it is clear
what the branch brings to ``Pyxel``. You can have many "shiny-new-features"
and switch in between them using the ``git checkout`` command.

To update this branch, you need to retrieve the changes from the master branch::

    git fetch upstream
    git rebase upstream/master

This will replay your commits on top of the latest ``Pyxel`` git master.  If this
leads to merge conflicts, you must resolve these before submitting your pull
request.  If you have uncommitted changes, you will need to ``git stash`` them
prior to updating.  This will effectively store your changes and they can be
reapplied after updating.

.. _contributing.documentation:

Contributing to the documentation
=================================

If you're not the developer type, contributing to the documentation is still of
huge value. You don't even have to be an expert on *``Pyxel`` to do so! In fact,
there are sections of the docs that are worse off after being written by
experts. If something in the docs doesn't make sense to you, updating the
relevant section after you figure it out is a great way to ensure it will help
the next person.


About the ``Pyxel`` documentation
---------------------------------

The documentation is written in **reStructuredText**, which is almost like writing
in plain English, and built using `Sphinx <http://sphinx.pocoo.org/>`__. The
Sphinx Documentation has an excellent `introduction to reST
<http://sphinx.pocoo.org/rest.html>`__. Review the Sphinx docs to perform more
complex changes to the documentation as well.

Some other important things to know about the docs:

- The ``Pyxel`` documentation consists of two parts: the docstrings in the code
  itself and the docs in this folder ``pyxel/docs/``.

  The docstrings are meant to provide a clear explanation of the usage of the
  individual functions, while the documentation in this folder consists of
  tutorial-like overviews per topic together with some other information
  (what's new, installation, etc).

- The docstrings follow the **Numpy Docstring Standard**, which is used widely
  in the Scientific Python community. This standard specifies the format of
  the different sections of the docstring. See `this document
  <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
  for a detailed explanation, or look at some of the existing functions to
  extend it in a similar manner.

- The tutorials make heavy use of the `ipython directive
  <http://matplotlib.org/sampledoc/ipython_directive.html>`_ sphinx extension.
  This directive lets you put code in the documentation which will be run
  during the doc build. For example::

      .. ipython:: python

          x = 2
          x**3

  will be rendered as::

      In [1]: x = 2

      In [2]: x**3
      Out[2]: 8

  Almost all code examples in the docs are run (and the output saved) during the
  doc build. This approach means that code examples will always be up to date,
  but it does make the doc building a bit more complex.

- Our API documentation in ``docs/api.rst`` houses the auto-generated
  documentation from the docstrings. For classes, there are a few subtleties
  around controlling which methods and attributes have pages auto-generated.

  Every method should be included in a ``toctree`` in ``api.rst``, else Sphinx
  will emit a warning.




Adding new models
--------------------

See the :ref:`Adding new models <new_model>` page.

Requirements
--------------------

**Testing:**

* pytest
* pytest-html
* pytest-cov
* pytest-pylint
* flake8
* flake8-docstrings
* pep8-naming
* mypy

**Create documentation:**

* doc8
* sphinx
* sphinx-rtd-theme
* sphinxcontrib-bibtex

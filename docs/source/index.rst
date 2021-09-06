.. _index:

========================================
Welcome to the **Pyxel** documentation !
========================================

**Version:** |version|

Pyxel is a general detector simulation framework.

An easy-to-use framework that can simulate a variety of imaging detector
effects combined on images (e.g. radiation and optical effects, noises)
made by CCD or CMOS-based detectors.


How the documentation is organized
==================================

A high-level overview of how the documentation is organized will help you known where
to look for certain things:

* :ref:`Tutorials<Tutorial Overview>` take you by the hand through a series of steps
  to install and to use Pyxel. **Start here if you're new to Pyxel**.

* :ref:`Background<Background Overview>` discuss key topics and concepts at a fairly
  high level and provide useful background information and explanation.

* :ref:`Reference guides<Reference Overview>` contain technical reference for APIs and
  other aspects of Pyxel. They describe how it works and how to use it but assume that
  you have a basic understanding of key concepts.

* :ref:`How-to guides<How-to Overview>` are recipes. They guide you through the steps
  involved in addressing the key problems and use-cases.
  They are mmore advanced than tutorials and assume some knownledge of how Pyxel works.


.. _Tutorial Overview:

Getting Started / Tutorials
===========================

**Tutorials**. Learn about the Pyxel's concepts.
Are you new to Pyxel ? This is the place to start !

.. toctree::
   :caption: Getting Started / Tutorials
   :maxdepth: 2
   :hidden:

   tutorials/overview.rst
   tutorials/install.rst
   tutorials/running.rst
   tutorials/examples.rst

* :doc:`tutorials/install`
* :doc:`tutorials/running`
* :doc:`tutorials/examples`

**Before you do anything else, start here** at our :doc:`tutorials/install` tutorial.


.. _How-to Overview:

How-to guides
=============

**Step-by-step guides**. Covers key tasks and operations and common problems.
These how-to guides are intended as recipes to solve common problems/tasks using Pyxel.

They are composed of the following **goal-oriented** series of steps:

* Working in the local development environment
* :doc:`howto/new_model`


.. note::

    We assume that the user has already some basic knowledge and understanding of Pyxel.

    If you are a beginner, you should have a look at the tutorials.

    A how-to guide is an answer to a question that only a user with some
    experience could even formulate and it offers a get-it-done information.

.. toctree::
   :caption: How-to guides
   :maxdepth: 1
   :hidden:

   howto/overview.rst
   howto/new_model.rst


.. _Background Overview:

Background / Explanations
=========================

**Explanations**.
Explanation of key concepts, best practices and techniques in Pyxel.

.. toctree::
   :caption: Background / Explanations
   :maxdepth: 1
   :hidden:

   background/overview.rst
   background/pipelines.rst
   background/detectors.rst
   background/data_structure.rst
   background/yaml.rst
   background/running_modes.rst

* :doc:`background/pipelines`
* :doc:`background/detectors`
* :doc:`background/data_structure`
* :doc:`background/yaml`
* :doc:`background/running_modes`


.. _Reference Overview:

Reference
=========

**Technical reference**.
Cover tools, components and resources.

.. toctree::
   :caption: Reference
   :maxdepth: 1
   :hidden:

   references/overview.rst
   references/apireference.rst
   references/models.rst
   references/contributing.rst
   references/changelog.rst

* :doc:`references/apireference`
* :doc:`references/models`
* :doc:`references/contributing`
* :doc:`references/changelog`


.. toctree::
   :caption: Developer Guide
   :maxdepth: 1
   :hidden:


.. toctree::
   :caption: About
   :maxdepth: 1
   :hidden:

   about/citation.rst
   about/contributors.rst
   about/license.rst
   about/faq.rst
   about/acronyms.rst
   about/acknowledgements.rst
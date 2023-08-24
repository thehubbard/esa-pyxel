.. _index:

========================================
Welcome to the **Pyxel** documentation !
========================================

Pyxel is a general detector simulation framework.

An easy-to-use framework that can simulate a variety of imaging detector
effects combined on images (e.g. radiation and optical effects, noises)
made by :term:`CCD` or :term:`CMOS`-based detectors.

**Version:** |version|

**Useful links**:
`Webpage and blog <https://esa.gitlab.io/pyxel/>`__ |
`Source repository <https://gitlab.com/esa/pyxel>`__ |
`Example repository <https://gitlab.com/esa/pyxel-data>`__


How the documentation is organized
==================================

A high-level overview of how the documentation is organized will help you known where
to look for certain things:

.. panels::
    :card: + intro-card text-center
    :column: col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex p-3
    :body: bg-white
    :header: bg-white
    :footer: bg-white

    ---

    **Getting started**
    ^^^^^^^^^^^^^^^^^^^

    Getting started section takes you by the hand through a series of steps on how
    to install and how to use Pyxel. Contains a tutorial and multiple examples.
    **Start here if you're new to Pyxel**.

    +++

    .. link-button:: introduction
            :type: ref
            :text: Get started
            :classes: btn-outline-primary btn-block stretched-link

    ---

    **How-to guides**
    ^^^^^^^^^^^^^^^^^

    How-to guides are recipes. They guide you through the steps
    involved in addressing the key problems and use-cases.
    They are more advanced than tutorials and assume some knowledge of how Pyxel works.

    +++

    .. link-button:: howtos
            :type: ref
            :text: To the how-to guides
            :classes: btn-outline-primary btn-block stretched-link

    ---

    **Background**
    ^^^^^^^^^^^^^^

    Background section discusses and explains key topics and concepts at a fairly
    high level and provides useful background information.

    +++

    .. link-button:: background
            :type: ref
            :text: To the background guide
            :classes: btn-outline-primary btn-block stretched-link

    ---

    **Reference**
    ^^^^^^^^^^^^^

    Reference guides contain technical reference for APIs, models and
    other aspects of Pyxel. They describe how it works and how to use it but assume that
    you have a basic understanding of key concepts.

    +++

    .. link-button:: reference
            :type: ref
            :text: To the reference guide
            :classes: btn-outline-primary btn-block stretched-link

.. toctree::
   :caption: Getting Started
   :maxdepth: 2
   :hidden:

   tutorials/overview.rst
   tutorials/install.rst
   tutorials/environments.rst
   tutorials/get_help.rst
   tutorials/running.rst
   tutorials/examples.rst

.. toctree::
   :caption: How-to guides
   :maxdepth: 1
   :hidden:

   howto/overview.rst
   howto/new_model.rst
   howto/detector_import_export.ipynb
   howto/json_schema.md

.. toctree::
   :caption: Background / Explanations
   :maxdepth: 2
   :hidden:

   background/overview.rst
   background/architecture.rst
   background/pixel_coordinate_conventions.rst
   background/detectors.rst
   background/pipeline.rst
   background/yaml.rst
   background/running_modes.rst

.. toctree::
   :caption: Reference
   :maxdepth: 1
   :hidden:

   references/overview.rst
   references/apireference.rst
   references/models.rst
   references/contributing.rst
   references/changelog.md

.. toctree::
   :caption: About
   :maxdepth: 1
   :hidden:

   about/brief_history.rst
   about/citation.rst
   about/contributors.rst
   about/license.rst
   about/FAQ.md
   about/acronyms.rst
   about/acknowledgements.rst
   about/bibliography.rst




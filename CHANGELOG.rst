Changelog
=========


version 0.7 / 2020-MM-DD
------------------------

Core
~~~~

* Update .gitignore file.
  (See `!123 <https://gitlab.com/esa/pyxel/-/merge_requests/123>`_).
* Added capability to load more image formats and tests.
  (See `!113 <https://gitlab.com/esa/pyxel/-/merge_requests/113>`_).
* Create a function 'pyxel.show_versions().
  (See `!114 <https://gitlab.com/esa/pyxel/-/merge_requests/114>`_).
* Shorter path to import/reference the models.
  (See `!131 <https://gitlab.com/esa/pyxel/-/merge_requests/131>`_).
* Remove deprecated methods from Photon class.
  (See `!119 <https://gitlab.com/esa/pyxel/-/merge_requests/119>`_).

Documentation
~~~~~~~~~~~~~

* Remove comments for magic methods.
  (See `!134 <https://gitlab.com/esa/pyxel/-/merge_requests/134>`_).



version 0.6 / 2020-09-16
------------------------

* Improved contributing guide
  (See `#68 <https://gitlab.com/esa/pyxel/issues/68>`_).
* Remove file '.gitlab-ci-doc.yml'
  (See `#73 <https://gitlab.com/esa/pyxel/issues/73>`_).
* Change license and add copyrights to all source files.
  (See `#69 <https://gitlab.com/esa/pyxel/issues/69>`_).
* Fix issues with example file 'examples/calibration_CDM_beta.yaml'.
  (See `#75 <https://gitlab.com/esa/pyxel/issues/75>`_).
* Fix issues with example file 'examples/calibration_CDM_irrad.yaml'.
  (See `#76 <https://gitlab.com/esa/pyxel/issues/76>`_).
* Updated Jupyter notebooks examples.
  (See `#87 <https://gitlab.com/esa/pyxel/issues/87>`_).
* Apply command 'isort' to the code base.
* Refactor class `ParametricPlotArgs`.
  (See `#77 <https://gitlab.com/esa/pyxel/issues/77>`_).
* Create class `SinglePlot`.
  (See `#78 <https://gitlab.com/esa/pyxel/issues/78>`_).
* Create class `CalibrationPlot`.
  (See `#79 <https://gitlab.com/esa/pyxel/issues/79>`_).
* Create class `ParametricPlot`.
  (See `#80 <https://gitlab.com/esa/pyxel/issues/80>`_).
* Add templates for bug report, feature request and merge request.
  (See `#105 <https://gitlab.com/esa/pyxel/issues/105>`_).
* Parallel computing for 'parametric' mode.
  (See `#111 <https://gitlab.com/esa/pyxel/issues/111>`_).
* Improved docker image.
  (See `#96 <https://gitlab.com/esa/pyxel/issues/96>`_).
* Fix calibration pipeline.
  (See `#113 <https://gitlab.com/esa/pyxel/issues/113>`_).
* CI/CD pipeline 'licenses-latests' fails.
  (See `#125 <https://gitlab.com/esa/pyxel/issues/125>`_).


version 0.5 / 2019-12-20
------------------------

* Clean-up code.
* Remove any dependencies to esapy_config
  (See `#54 <https://gitlab.com/esa/pyxel/issues/54>`_).
* Refactor charge generation models to avoid code duplication
  (See `#49 <https://gitlab.com/esa/pyxel/issues/49>`_).
* Implement multi-threaded/multi-processing mode
  (See `#44 <https://gitlab.com/esa/pyxel/issues/44>`_).


version 0.4 / 2019-07-09
------------------------

* Running modes implemented:
  * Calibration mode for model fitting and detector optimization
  * Dynamic mode for time-dependent (destructive and non-destructive) detector readout
  * Parallel option for Parametric mode
* Models added:
  * CDM Charge Transfer Inefficiency model
  * POPPY physical optical propagation model
  * SAR ADC signal digitization model
* Outputs class for post-processing and saving results
* Logging, setup and versioneer
* Examples
* Documentation

version 0.3 / 2018-03-26
------------------------

* Single and Parametric mode have been implemented
* Infrastructure code has been placed in 2 new projects: esapy_config and esapy_web
* Web interface (GUI) is dynamically generated based on attrs definitions
* NGHxRG noise generator model has been added

version 0.2 / 2018-01-18
------------------------

* TARS cosmic ray model has been reimplemented and added

version 0.1 / 2018-01-10
------------------------

* Prototype: first pipeline for a CCD detector

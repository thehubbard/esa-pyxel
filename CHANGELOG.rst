Changelog
=========


version 0.8 UNRELEASED/ 2020-MM-DD
----------------------------------

Core
~~~~

* Improved user friendliness.
  (See `#144 <https://gitlab.com/esa/pyxel/issues/144>`_).
* Simplified the look of YAML configuration files.
  (See `#118 <https://gitlab.com/esa/pyxel/issues/118>`_).
* Extracted functions to run modes separately from pyxel.run.run()
  (See `#61 <https://gitlab.com/esa/pyxel/issues/61>`_).
* Refactored YAML loader, returns a class Configuration instead of a dictionary.
  (See `#60 <https://gitlab.com/esa/pyxel/issues/60>`_).
* Created new classes Single and Dynamic to store running mode parameters.
  (See `#121 <https://gitlab.com/esa/pyxel/issues/121>`_).
* Split class Outputs for different modes and moved to inputs_ouputs.
  (See `#149 <https://gitlab.com/esa/pyxel/issues/149>`_).
* Added a simple Inter Pixel Capacitance model for CMOS detectors.
  (See `#65 <https://gitlab.com/esa/pyxel/issues/65>`_).
* Added a model for the amplifier crosstalk.
  (See `#116 <https://gitlab.com/esa/pyxel/issues/116>`_).
* Added ability to load custom QE maps.
  (See `#117 <https://gitlab.com/esa/pyxel/issues/117>`_).

Others
~~~~~~

* Change licence to MIT.
  (See `!142 <https://gitlab.com/esa/pyxel/-/merge_requests/142>`_).
* Change Pyxel's package name to 'pyxel-sim'.
  (See `!144 <https://gitlab.com/esa/pyxel/-/merge_requests/114>`_).
* Added a 'How to release' guide.
  (See `#109 <https://gitlab.com/esa/pyxel/issues/109>`_).
* Remove_folder_examples_data.
  (See `!148 <https://gitlab.com/esa/pyxel/-/merge_requests/148>`_).
* Fix typo in documentation.
  (See `!149 <https://gitlab.com/esa/pyxel/-/merge_requests/149>`_).


version 0.7 / 2020-10-22
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
  (See `!126 <https://gitlab.com/esa/pyxel/-/merge_requests/126>`_).
* Remove deprecated methods from Photon class.
  (See `!119 <https://gitlab.com/esa/pyxel/-/merge_requests/119>`_).
* Instances of 'DetectionPipeline' are not serializable.
  (See `!120 <https://gitlab.com/esa/pyxel/-/merge_requests/120>`_).
* Cannot run 'calibration' pipeline with multiprocessing or ipyparallel islands.
  (See `!121 <https://gitlab.com/esa/pyxel/-/merge_requests/121>`_).
* Make package and script 'pyxel' executable.
  (See `!112 <https://gitlab.com/esa/pyxel/-/merge_requests/112>`_).
* Created a function inputs_outputs.load_table().
  (See `!132 <https://gitlab.com/esa/pyxel/-/merge_requests/132>`_).
* Reimplement convolution in POPPY optical_psf model.
  (See `#52 <https://gitlab.com/esa/pyxel/issues/52>`_).
* Add property 'Detector.numbytes' and/or method 'Detector.memory_usage()'
  (See `!116 <https://gitlab.com/esa/pyxel/-/merge_requests/116>`_).
* Created jupyxel.py for jupyter notebook visualization.
  (See `!122 <https://gitlab.com/esa/pyxel/-/merge_requests/122>`_).

Documentation
~~~~~~~~~~~~~

* Remove comments for magic methods.
  (See `!127 <https://gitlab.com/esa/pyxel/-/merge_requests/127>`_).


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

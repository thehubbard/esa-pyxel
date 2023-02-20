# Frequently Asked Questions
- [How to contact the developers of Pyxel?](#how-to-contact-the-developers-of-pyxel)
- [How to install Pyxel?](#how-to-install-pyxel)
- [How to run Pyxel?](#how-to-run-pyxel)
- [Is there a simple example of a configuration file?](#is-there-a-simple-example-of-a-configuration-file)
- [Is there a way to add photons to a Detector directly from a file containing photons or pixel instead of converting from image in ADU to photons via the PTF?](#is-there-a-way-to-add-photons-to-a-detector-directly-from-a-file-containing-photons-or-pixel-instead-of-converting-from-image-in-adu-to-photons-via-the-ptf)
- [Is there any way use Pyxel to produce a bias or dark image without including any image file?](#is-there-any-way-use-pyxel-to-produce-a-bias-or-dark-image-without-including-any-image-file)
- [What are the different running modes in Pyxel?](#what-are-the-different-running-modes-in-pyxel)
- [What detectors types are implemented in Pyxel?](#what-detectors-types-are-implemented-in-pyxel)
- [What is the easiest way to get the signal to noise ratio from the detector data buckets?](#what-is-the-easiest-way-to-get-the-signal-to-noise-ratio-from-the-detector-data-buckets)
- [Where can I see the latest changes in Pyxel?](#where-can-i-see-the-latest-changes-in-pyxel)
- [Why do I retrieve a blank image but no error?](#why-do-i-retrieve-a-blank-image-but-no-error)

<a name="how-to-contact-the-developers-of-pyxel"></a>
## How to contact the developers of Pyxel?

If you found a bug or want to suggest a new feature, you can create an [issue on Gitlab](https://gitlab.com/esa/pyxel/-/issues).

If you have a question, you can use the Chat on [Gitter](https://gitter.im/pyxel-framework/community) to get help from the Pyxel community.

[Read more](https://esa.gitlab.io/pyxel/doc/stable/tutorials/get_help.html).

If you are using Pyxel on a regular basis and want to [contribute](http://localhost:52873/references/contributing.html), let us know.

You can always reach us via email: [pyxel@esa.int](mailto:pyxel@esa.int).

<a name="how-to-install-pyxel"></a>
## How to install Pyxel?

Look at the [Installation Guide](https://esa.gitlab.io/pyxel/doc/stable/tutorials/install.html).

<a name="how-to-run-pyxel"></a>
## How to run Pyxel?

Look in the documentation to know [how to run Pyxel](https://esa.gitlab.io/pyxel/doc/stable/tutorials/running.html).

<a name="is-there-a-simple-example-of-a-configuration-file"></a>
## Is there a simple example of a configuration file?

There are a couple of models that are required in the pipeline to get an image in the end.  
One have to make use of simple models in the pipeline that the conversion **photon->charge->pixel->signal->image** is happening.

A simple pipeline example of a configuration yaml file in exposure mode can be found here: 
[simple_exposure.yaml](https://gitlab.com/esa/pyxel-data/-/blob/master/examples/exposure/simple_exposure.yaml).

<a name="is-there-a-way-to-add-photons-to-a-detector-directly-from-a-file-containing-photons-or-pixel-instead-of-converting-from-image-in-adu-to-photons-via-the-ptf"></a>
## Is there a way to add photons to a Detector directly from a file containing photons or pixel instead of converting from image in ADU to photons via the PTF?

Yes, with the model 'load_image' in the photon generation model group it is possible to load photons directly from a file.
You can set the argument 'convert_to_photons' to false, and it will use your input array without converting it via PTF.
See [here](https://esa.gitlab.io/pyxel/doc/stable/references/model_groups/photon_collection_models.html#load-image) for more details.

<a name="is-there-any-way-use-pyxel-to-produce-a-bias-or-dark-image-without-including-any-image-file"></a>
## Is there any way use Pyxel to produce a bias or dark image without including any image file?

Without the models in the pipeline, Pyxel will still run, but will generate nothing, so just zero arrays. 
Dark image for example would be generated using a dark current model in charge_generation. 
The detector object stores the data and some properties that are needed by more than one model, 
but it doesn't directly influence how the stored data is edited, this information is in the pipeline.

<a name="what-are-the-different-running-modes-in-pyxel"></a>
## What are the different running modes in Pyxel?

There are three [running modes](https://esa.gitlab.io/pyxel/doc/stable/background/running_modes.html) in Pyxel:

[Exposure mode:](https://esa.gitlab.io/pyxel/doc/stable/background/running_modes/exposure_mode.html#exposure-mode) is used for a simulation of a single exposure, at a single or with incrementing readout times 
(quick look/ health check, simulation of non-destructive readout mode and time-dependent effects).
[Observation mode:](https://esa.gitlab.io/pyxel/doc/stable/background/running_modes/observation_mode.html) consists of multiple exposure pipelines looping over a range of model or detector parameters (sensitivity analysis).
[Calibration mode:](https://esa.gitlab.io/pyxel/doc/stable/background/running_modes/calibration_mode.html) is used to optimize model or detector parameters to fit target data sets using a user-defined fitness function/figure of merit 
(model fitting, instrument optimization).

<a name="what-detectors-types-are-implemented-in-pyxel"></a>
## What detectors types are implemented in Pyxel?

The following [detector types](https://esa.gitlab.io/pyxel/doc/stable/background/detectors.html#implemented-detector-types) 
are implemented in Pyxel:
- [CCD](https://esa.gitlab.io/pyxel/doc/stable/background/detectors/ccd.html)
- [CMOS](https://esa.gitlab.io/pyxel/doc/stable/background/detectors/cmos.html)
- [MKID](https://esa.gitlab.io/pyxel/doc/stable/background/detectors/mkid.html)
- [APD](https://esa.gitlab.io/pyxel/doc/stable/background/detectors/apd.html)

<a name="what-is-the-easiest-way-to-get-the-signal-to-noise-ratio-from-the-detector-data-buckets"></a>
## What is the easiest way to get the signal to noise ratio from the detector data buckets?

The easiest way is like this:

```python 
signal = result.signal.mean()
noise = result.signal.var()
snr = signal / noise
snr
```

The snr is an array with each exposure time in exposure mode 
(ndarray when using observation mode) with the result of the simulation, e.g. in exposure mode:
```python
result = pyxel.exposure_mode(
exposure=exposure,
detector=detector, 
pipeline=pipeline
)
```

<a name="where-can-i-see-the-latest-changes-in-pyxel"></a>
## Where can I see the latest changes in Pyxel?

The latest changes can be found in the [Changelog file](https://esa.gitlab.io/pyxel/doc/stable/references/changelog.html).

<a name="why-do-i-retrieve-a-blank-image-but-no-error"></a>
## Why do I retrieve a blank image but no error?

There are a couple of models that are required in the pipeline to get an image in the end.  
One have to make use of simple models in the pipeline that the conversion **photon->charge->pixel->signal->image** is happening.

Example: 
If you have only the model "load_image" in your pipeline and make use of the function "pyxel.display_detector(detector)" 
you will retrieve the plot with photon, but the plots showing pixel or image are blank, because no conversion is taking place in the pipeline.

A simple pipeline example of a configuration yaml file in exposure mode can be found here: 
[simple_exposure.yaml](https://gitlab.com/esa/pyxel-data/-/blob/master/examples/exposure/simple_exposure.yaml).

<hr>

Generated by [FAQtory](https://github.com/willmcgugan/faqtory)

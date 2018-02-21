from pathlib import Path
# import typing as t
import numpy as np
import pytest
from astropy.io import fits

# from pyxel.pipelines.processor import Processor
# from pyxel.pipelines.ccd_pipeline import CCDDetectionPipeline
# from pyxel.pipelines.cmos_pipeline import CMOSDetectionPipeline
# from pyxel.detectors.ccd import CCD
from pyxel.io.yaml_processor import load_config


@pytest.mark.parametrize("input_filename, exp_filename", [
    ('tests/data/pipeline_tars.yaml', 'tests/data/expected_ccd_pipeline01.fits'),
    ('tests/data/pipeline_tars_no_tags.yaml', 'tests/data/expected_ccd_pipeline01.fits'),
])
def test_pipeline_tars(input_filename, exp_filename):
    np.random.seed(19690906)

    # Step 1: Get the pipeline configuration
    cfg = load_config(Path(input_filename))
    processor = cfg['ccd_process']      # type: pyxel.pipelines.processor.Processor

    pipeline = processor.pipeline  # type: t.Union[CCDDetectionPipeline, CMOSDetectionPipeline]

    # Step 2: Run the pipeline
    detector = pipeline.run_pipeline(processor.detector)  # type: CCD
    print('Pipeline completed.')

    expected = fits.getdata(exp_filename)
    image = detector.image  # type: np.ndarray

    # fits.writeto('tests/data/result.fits', image, None, output_verify='ignore', overwrite=True)
    assert isinstance(image, np.ndarray)
    np.testing.assert_array_equal(image, expected)

"""TBW."""
from pyxel import util
from pyxel.pipelines.detector_pipeline import DetectionPipeline
from pyxel.detectors.detector import Detector
from pyxel.physics.charge import Charge
from pyxel.physics.photon import Photon
from pyxel.physics.pixel import Pixel
from pyxel.pipelines.models import Models


class CCDDetectionPipeline(DetectionPipeline):
    """TBW."""

    def __init__(self,
                 charge_transfer: Models,
                 **kwargs) -> None:
        """TBW.

        :param charge_transfer:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.charge_transfer = charge_transfer

        self._name = 'ccd'
        self._model_groups = ['photon_generation',
                              'optics',
                              'charge_generation',
                              'charge_collection',
                              'charge_transfer',
                              'charge_measurement',
                              'readout_electronics']

        self._model_steps = {
            'photon_generation':    ['load_image', 'photon_level',
                                     'shot_noise'],
            'optics':               [],
            'charge_generation':    ['photoelectrons',
                                     'tars'],
            'charge_collection':    ['fixed_pattern_noise',
                                     'full_well'],
            'charge_transfer':      ['cdm'],
            'charge_measurement':   ['output_node_noise'],
            'readout_electronics':  []
        }

    def copy(self):
        """TBW."""
        return CCDDetectionPipeline(**util.copy_state(self))

    def __getstate__(self):
        """TBW."""
        kwargs = super().__getstate__()
        kwargs_obj = {
            'charge_transfer': self.charge_transfer
        }
        return {**kwargs, **kwargs_obj}

    def run_pipeline(self, detector: Detector) -> Detector:
        """TBW.

        :param detector:
        :return:
        """
        # INITIALIZATION (open or generate image):
        # START -> create photons ->
        detector.photons = Photon(detector)
        # detector.photons.generate_photons()
        detector = self.run_model_group('photon_generation', detector)

        # OPTICS:
        # -> transport/modify photons ->
        detector = self.run_model_group('optics', detector)

        # CHARGE GENERATION:
        # -> create charges & remove photons ->
        detector.charges = Charge(detector)
        detector = self.run_model_group('charge_generation', detector)

        # CHARGE COLLECTION:
        # -> transport/modify charges ->
        # -> collect charges in pixels ->
        detector.pixels = Pixel(detector)
        detector.pixels.generate_pixels()
        detector = self.run_model_group('charge_collection', detector)

        # CHARGE TRANSFER:
        # -> transport/modify pixels ->
        detector = self.run_model_group('charge_transfer', detector)

        # CHARGE READOUT
        # -> create signal -> modify signal ->
        detector.signal = detector.pixels.generate_2d_charge_array()
        detector.signal = detector.signal.astype('float64')

        detector = self.run_model_group('charge_measurement', detector)

        # READOUT ELECTRONICS
        # -> create image -> modify image -> END
        detector = self.run_model_group('readout_electronics', detector)

        return detector
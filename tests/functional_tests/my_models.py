# from pyxel import MetaModel
# from pyxel import registry
from pyxel.detectors.detector import Detector
from pyxel.pipelines.model_registry import registry
from pyxel import register

# from pyxel.util import objmod as om


@register('charge_generation', name='my_class_model')
class MyClassModel:

    def __call__(self, detector: Detector, level: int, noise: float=2.0):
        setattr(detector, 'level', level)
        setattr(detector, 'noise', noise)
        return detector


@register('charge_generation', name='my_other_class_model')
class MyOtherClassModel:

    def __call__(self, detector: Detector, std: float=2.0):
        setattr(detector, 'std', std)
        return detector


# class MyClassModel(metaclass=MetaModel,
#                    name='my_class_model',
#                    group='charge_generation'):
#
#     def __call__(self, detector: Detector, level: int, noise: float=2.0):
#         setattr(detector, 'level', level)
#         setattr(detector, 'noise', noise)
#         return detector
#
#
# class MyOtherClassModel(metaclass=MetaModel,
#                         name='my_other_class_model',
#                         group='charge_generation'):
#
#     def __call__(self, detector: Detector, std: float=2.0):
#         setattr(detector, 'std', std)
#         return detector


def my_function_model(detector: Detector, level, noise: float=2.0):
    # set a new attribute so it can be checked later
    setattr(detector, 'level', level)
    setattr(detector, 'noise', noise)
    return detector


register('charge_generation', my_function_model)


@registry.decorator(group='charge_generation', name='my_dec_model_class')
class MyDecoratedModel:

    def __call__(self, detector: Detector, class_std=1.0):
        setattr(detector, 'class_std', class_std)
        return detector


@registry.decorator(group='charge_generation', name='my_dec_model_func')
def my_decorated_function(detector: Detector, func_std=2.0):
    setattr(detector, 'func_std', func_std)
    return detector


registry_map = {
    'photon_generation': [
        {
            'func': 'pyxel.models.photon_generation.load_image',
        },
        {
            'func': 'pyxel.models.photon_generation.add_photon_level',
        },
        {
            'func': 'pyxel.models.photon_generation.add_shot_noise',

        }
    ],
    'optics': [

    ],
    'charge_generation': [
        {
            'func': 'pyxel.models.photoelectrons.simple_conversion',
            'name': 'photoelectrons',
        },
        {
            'func': 'pyxel.models.tars.tars.run_tars',
            'name': 'tars'
        }
    ],
    'charge_collection': [
        {
            'func': 'pyxel.models.ccd_noise.add_fix_pattern_noise',
            'type': 'ccd',
        },
        {
            'func': 'pyxel.models.full_well.simple_pixel_full_well',
            'name': 'full_well',
        }
    ],
    'charge_transfer': [
        {
            'func': 'pyxel.models.cdm.CDM.cdm',
            'type': 'ccd',
        }
    ],
    'charge_measurement': [
        {
            'func': 'pyxel.models.ccd_noise.add_output_node_noise',
            'type': 'ccd',
        },
        {
            'func': 'pyxel.models.nghxrg.nghxrg.ktc_bias_noise',
            'type': 'cmos',
        },
        {
            'func': 'pyxel.models.nghxrg.nghxrg.white_read_noise',
        }
    ],
    'signal_transfer': [
        {
            'func': 'pyxel.models.nghxrg.nghxrg.corr_pink_noise',
            'type': 'cmos',
        },
        {
            'func': 'pyxel.models.nghxrg.nghxrg.uncorr_pink_noise',
            'type': 'cmos',
        },
        {
            'func': 'pyxel.models.nghxrg.nghxrg.acn_noise',
            'type': 'cmos',
        }

    ],
    'readout_electronics': [
        {
            'func': 'pyxel.models.nghxrg.nghxrg.pca_zero_noise',
            'type': 'cmos',
        }
    ]
}

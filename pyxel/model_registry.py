"""Models defined in the pyxel library.

A user may copy this and add their own models or remove
existing ones. If auto-registration of a model is not
used, then the registry map below may be used to explicitly
define which model is available and which order they are to
be executed.
"""

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

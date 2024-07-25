#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyxel.detectors import CCD, CCDGeometry, Characteristics, Detector, Environment
from pyxel.models.charge_generation import cosmix


@pytest.fixture
def ccd_8x8() -> CCD:
    detector = CCD(
        geometry=CCDGeometry(
            row=8,
            col=8,
            pixel_horz_size=10.0,
            pixel_vert_size=10.0,
            total_thickness=40.0,
        ),
        environment=Environment(),
        characteristics=Characteristics(),
    )
    detector.set_readout(times=[1.0, 5.0, 7.0], non_destructive=False)

    return detector


@pytest.mark.parametrize(
    "extra_params",
    [
        pytest.param({}, id="without extra parameters"),
        pytest.param(
            {"incident_angles": None, "starting_position": None},
            id="with empties 'incident_angles' and 'starting_positions'",
        ),
        pytest.param(
            {
                "stepsize": [
                    {
                        "type": "proton",
                        "energy": 100.0,
                        "thickness": 40.0,
                        "filename": "{folder}/data/stepsize_proton_100MeV_40um_Si_10k.ascii",
                    },
                    {
                        "type": "proton",
                        "energy": 100.0,
                        "thickness": 50.0,
                        "filename": "{folder}/data/stepsize_proton_100MeV_50um_Si_10k.ascii",
                    },
                    {
                        "type": "proton",
                        "energy": 100.0,
                        "thickness": 60.0,
                        "filename": "{folder}/data/stepsize_proton_100MeV_60um_Si_10k.ascii",
                    },
                    {
                        "type": "proton",
                        "energy": 100.0,
                        "thickness": 70.0,
                        "filename": "{folder}/data/stepsize_proton_100MeV_70um_Si_10k.ascii",
                    },
                    {
                        "type": "proton",
                        "energy": 100.0,
                        "thickness": 100.0,
                        "filename": "{folder}/data/stepsize_proton_100MeV_100um_Si_10k.ascii",
                    },
                ]
            },
            id="with 'stepsize'",
        ),
    ],
)
def test_cosmix_stepsize(extra_params, ccd_8x8: CCD, request: pytest.FixtureRequest):
    detector: Detector = ccd_8x8
    assert isinstance(extra_params, dict)

    charge_2d = np.array(
        [
            [14323.0, 13206.0, 13087.0, 13152.0, 13165.0, 13138.0, 13106.0, 13162.0],
            [13284.0, 11870.0, 11779.0, 11877.0, 11887.0, 11853.0, 11797.0, 11797.0],
            [13231.0, 11856.0, 11788.0, 11875.0, 11892.0, 11859.0, 11789.0, 11715.0],
            [13331.0, 11971.0, 11861.0, 11946.0, 11975.0, 11914.0, 11842.0, 11776.0],
            [13387.0, 12004.0, 11868.0, 11969.0, 11963.0, 11816.0, 11740.0, 11765.0],
            [13416.0, 12022.0, 11887.0, 11977.0, 11894.0, 11659.0, 11592.0, 11727.0],
            [13404.0, 12034.0, 11930.0, 11988.0, 11829.0, 11558.0, 11508.0, 11686.0],
            [13358.0, 12026.0, 11948.0, 11952.0, 11755.0, 11513.0, 11443.0, 11536.0],
        ]
    )
    detector.charge.add_charge_array(charge_2d)

    current_folder: Path = request.path.parent

    if "stepsize" in extra_params:
        new_stepsizes = [
            {
                key: value if key != "filename" else value.format(folder=current_folder)
                for key, value in element.items()
            }
            for element in extra_params["stepsize"]
        ]

        extra_params["stepsize"] = new_stepsizes

    cosmix(
        detector=detector,
        simulation_mode="cosmic_ray",
        running_mode="stepsize",
        particle_type="proton",
        initial_energy=100.0,
        particles_per_second=100.0,
        spectrum_file=str(
            current_folder / "data/proton_L2_solarMax_11mm_Shielding.txt"
        ),
        seed=1234,
        progressbar=False,
        **extra_params,
    )

    new_charges_df = detector.charge.frame

    assert isinstance(new_charges_df, pd.DataFrame)
    assert len(new_charges_df) == 5901

    exp_head_df = pd.DataFrame(
        {
            "charge": [-1, -1, -1, -1, -1],
            "number": [14323.0, 13206.0, 13087.0, 13152.0, 13165.0],
            "init_energy": [0.0, 0.0, 0.0, 0.0, 0.0],
            "energy": [0.0, 0.0, 0.0, 0.0, 0.0],
            "init_pos_ver": [5.0, 5.0, 5.0, 5.0, 5.0],
            "init_pos_hor": [5.0, 15.0, 25.0, 35.0, 45.0],
            "init_pos_z": [0.0, 0.0, 0.0, 0.0, 0.0],
            "position_ver": [5.0, 5.0, 5.0, 5.0, 5.0],
            "position_hor": [5.0, 15.0, 25.0, 35.0, 45.0],
            "position_z": [0.0, 0.0, 0.0, 0.0, 0.0],
            "velocity_ver": [0.0, 0.0, 0.0, 0.0, 0.0],
            "velocity_hor": [0.0, 0.0, 0.0, 0.0, 0.0],
            "velocity_z": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    pd.testing.assert_frame_equal(new_charges_df.head(), exp_head_df)

    exp_tail_df = pd.DataFrame(
        {
            "charge": {5896: -1, 5897: -1, 5898: -1, 5899: -1, 5900: -1},
            "number": {5896: 57.0, 5897: 1.0, 5898: 1.0, 5899: 22.0, 5900: 1.0},
            "init_energy": {
                5896: 1000.0,
                5897: 1000.0,
                5898: 1000.0,
                5899: 1000.0,
                5900: 1000.0,
            },
            "energy": {
                5896: 1000.0,
                5897: 1000.0,
                5898: 1000.0,
                5899: 1000.0,
                5900: 1000.0,
            },
            "init_pos_ver": {
                5896: 51.1005849516724,
                5897: 50.37673409495956,
                5898: 48.22123983509702,
                5899: 45.73643756556448,
                5900: 44.30205495632059,
            },
            "init_pos_hor": {
                5896: 74.95341327530997,
                5897: 75.44758533887565,
                5898: 76.91913854147528,
                5899: 78.61550990541043,
                5900: 79.59476109697735,
            },
            "init_pos_z": {
                5896: -12.353512854409614,
                5897: -11.378000220104713,
                5898: -8.473103632406465,
                5899: -5.1244082689911785,
                5900: -3.1913327676386425,
            },
            "position_ver": {
                5896: 51.1005849516724,
                5897: 50.37673409495956,
                5898: 48.22123983509702,
                5899: 45.73643756556448,
                5900: 44.30205495632059,
            },
            "position_hor": {
                5896: 74.95341327530997,
                5897: 75.44758533887565,
                5898: 76.91913854147528,
                5899: 78.61550990541043,
                5900: 79.59476109697735,
            },
            "position_z": {
                5896: -12.353512854409614,
                5897: -11.378000220104713,
                5898: -8.473103632406465,
                5899: -5.1244082689911785,
                5900: -3.1913327676386425,
            },
            "velocity_ver": {5896: 0.0, 5897: 0.0, 5898: 0.0, 5899: 0.0, 5900: 0.0},
            "velocity_hor": {5896: 0.0, 5897: 0.0, 5898: 0.0, 5899: 0.0, 5900: 0.0},
            "velocity_z": {5896: 0.0, 5897: 0.0, 5898: 0.0, 5899: 0.0, 5900: 0.0},
        }
    )
    pd.testing.assert_frame_equal(new_charges_df.tail(), exp_tail_df, check_exact=False)

    new_charges_2d = detector.charge.array
    assert isinstance(new_charges_2d, np.ndarray)

    exp_charges_2d = np.array(
        [
            [19100.0, 15914.0, 17866.0, 17964.0, 22891.0, 15026.0, 16693.0, 20217.0],
            [18950.0, 20544.0, 17637.0, 24040.0, 14931.0, 18594.0, 24610.0, 13748.0],
            [26567.0, 21681.0, 24794.0, 20656.0, 15133.0, 19619.0, 20048.0, 20458.0],
            [20179.0, 20718.0, 14990.0, 17468.0, 14153.0, 18229.0, 16821.0, 21951.0],
            [21270.0, 18897.0, 22068.0, 18830.0, 22871.0, 16733.0, 16618.0, 17532.0],
            [14306.0, 25509.0, 23716.0, 24736.0, 16337.0, 24887.0, 25122.0, 16977.0],
            [24744.0, 28712.0, 17054.0, 18474.0, 17091.0, 24113.0, 13528.0, 12498.0],
            [16440.0, 21779.0, 16450.0, 14755.0, 17037.0, 13809.0, 13873.0, 13014.0],
        ]
    )
    np.testing.assert_almost_equal(new_charges_2d, exp_charges_2d)

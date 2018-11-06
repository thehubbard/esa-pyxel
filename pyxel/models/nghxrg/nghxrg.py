#   --------------------------------------------------------------------------
#   Copyright 2018 SCI-FIV, ESA (European Space Agency)
#   --------------------------------------------------------------------------
"""PyXel! wrapper class for NGHXRG - Teledyne HxRG Noise Generator model."""

# import logging
from os import path

import typing as t

# import numpy as np
from pyxel.detectors.cmos import CMOS
from pyxel.detectors.cmos_geometry import CMOSGeometry  # noqa: F401
from pyxel.detectors.ccd_geometry import CCDGeometry  # noqa: F401
from pyxel.detectors.geometry import Geometry  # noqa: F401
from pyxel.models.nghxrg.nghxrg_beta import HXRGNoise
from pyxel.pipelines.model_registry import registry


def get_geom(detector_obj):
    """TBW.

    :param detector_obj:
    :return:
    """
    if isinstance(detector_obj.geometry, CMOSGeometry):
        return t.cast(CMOSGeometry, detector_obj.geometry)
    elif isinstance(detector_obj.geometry, CCDGeometry):
        return t.cast(CCDGeometry, detector_obj.geometry)
    else:
        return t.cast(Geometry, detector_obj.geometry)


# def get_char(detector_obj):
#     if isinstance(detector_obj.characteristic, CMOSCharacteristic):
#         return t.cast(CMOSCharacteristic, detector_obj.characteristic)
#     if isinstance(detector_obj.characteristic, CCDCharacteristic):
#         return t.cast(CCDCharacteristic, detector_obj.characteristic)
#     else:
#         return t.cast(Characteristic, detector_obj.characteristic)


@registry.decorator('charge_measurement', name='nghxrg_ktc_bias', detector='cmos')
def ktc_bias_noise(detector: CMOS,
                   ktc_noise: float = None,
                   bias_offset: float = None,
                   bias_amp: float = None,
                   window_mode: str = 'FULL',
                   wind_x0: int = 0, wind_y0: int = 0,
                   wind_x_size: int = 0, wind_y_size: int = 0) -> CMOS:
    """TBW.

    :param detector:
    :param ktc_noise:
    :param bias_offset:
    :param bias_amp:
    :param window_mode:
    :param wind_x0:
    :param wind_y0:
    :param wind_x_size:
    :param wind_y_size:
    :return:
    """
    new_detector = detector
    geo = t.cast(CMOSGeometry, new_detector.geometry)

    number_of_fits = 1

    ng_h2rg = HXRGNoise(n_out=geo.n_output,
                        nroh=geo.n_row_overhead,
                        nfoh=geo.n_frame_overhead,
                        reverse_scan_direction=geo.reverse_scan_direction,
                        reference_pixel_border_width=geo.reference_pixel_border_width,
                        pca0_file=path.dirname(path.abspath(__file__)) + '/nirspec_pca0.fits',
                        det_size_x=geo.col, det_size_y=geo.row,
                        wind_mode=window_mode,
                        wind_x_size=wind_x_size, wind_y_size=wind_y_size,
                        wind_x0=wind_x0, wind_y0=wind_y0,
                        cube_z=number_of_fits,
                        verbose=True)

    result = ng_h2rg.add_ktc_bias_noise(ktc_noise=ktc_noise, bias_offset=bias_offset, bias_amp=bias_amp)

    result = ng_h2rg.format_result(result)

    if window_mode == 'FULL':
        new_detector.signal.array += result
    elif window_mode == 'WINDOW':
        new_detector.signal.array[wind_y0:wind_y0 + wind_y_size, wind_x0:wind_x0 + wind_x_size] += result
    # TODO: add to new_detector.signal.array OR new_detector.image OR charge dataframe ?

    # ng_h2rg.create_hdu(result, 'pyxel/hxrg_read_noise.fits')

    return new_detector


@registry.decorator('charge_measurement', name='nghxrg_read', detector='cmos')
def white_read_noise(detector: CMOS,
                     rd_noise: float = None,
                     ref_pixel_noise_ratio: float = None,
                     window_mode: str = 'FULL',
                     wind_x0: int = 0, wind_y0: int = 0,
                     wind_x_size: int = 0, wind_y_size: int = 0) -> CMOS:
    """TBW.

    :param detector:
    :param rd_noise:
    :param ref_pixel_noise_ratio:
    :param window_mode:
    :param wind_x0:
    :param wind_y0:
    :param wind_x_size:
    :param wind_y_size:
    :return:
    """
    new_detector = detector  # type: CMOS

    geo = get_geom(new_detector)

    number_of_fits = 1

    ng_h2rg = HXRGNoise(n_out=geo.n_output,
                        nroh=geo.n_row_overhead,
                        nfoh=geo.n_frame_overhead,
                        reverse_scan_direction=geo.reverse_scan_direction,
                        reference_pixel_border_width=geo.reference_pixel_border_width,
                        pca0_file=path.dirname(path.abspath(__file__)) + '/nirspec_pca0.fits',
                        det_size_x=geo.col, det_size_y=geo.row,
                        wind_mode=window_mode,
                        wind_x_size=wind_x_size, wind_y_size=wind_y_size,
                        wind_x0=wind_x0, wind_y0=wind_y0,
                        cube_z=number_of_fits,
                        verbose=True)

    result = ng_h2rg.add_white_read_noise(rd_noise=rd_noise, reference_pixel_noise_ratio=ref_pixel_noise_ratio)

    result = ng_h2rg.format_result(result)

    if window_mode == 'FULL':
        new_detector.signal.array += result
    elif window_mode == 'WINDOW':
        new_detector.signal.array[wind_y0:wind_y0 + wind_y_size, wind_x0:wind_x0 + wind_x_size] += result
    # TODO: add to new_detector.signal.array OR new_detector.image OR charge dataframe ?

    # ng_h2rg.create_hdu(result, 'pyxel/hxrg_read_noise.fits')

    return new_detector


@registry.decorator('signal_transfer', name='nghxrg_acn', detector='cmos')
def acn_noise(detector: CMOS,
              acn: float = None,
              window_mode: str = 'FULL',
              wind_x0: int = 0, wind_y0: int = 0,
              wind_x_size: int = 0, wind_y_size: int = 0) -> CMOS:
    """TBW.

    :param detector:
    :param acn:
    :param window_mode:
    :param wind_x0:
    :param wind_y0:
    :param wind_x_size:
    :param wind_y_size:
    :return:
    """
    new_detector = detector
    geo = t.cast(CMOSGeometry, new_detector.geometry)

    number_of_fits = 1

    ng_h2rg = HXRGNoise(n_out=geo.n_output,
                        nroh=geo.n_row_overhead,
                        nfoh=geo.n_frame_overhead,
                        reverse_scan_direction=geo.reverse_scan_direction,
                        reference_pixel_border_width=geo.reference_pixel_border_width,
                        pca0_file=path.dirname(path.abspath(__file__)) + '/nirspec_pca0.fits',
                        det_size_x=geo.col, det_size_y=geo.row,
                        wind_mode=window_mode,
                        wind_x_size=wind_x_size, wind_y_size=wind_y_size,
                        wind_x0=wind_x0, wind_y0=wind_y0,
                        cube_z=number_of_fits,
                        verbose=True)

    result = ng_h2rg.add_acn_noise(acn=acn)

    result = ng_h2rg.format_result(result)

    if window_mode == 'FULL':
        new_detector.signal.array += result
    elif window_mode == 'WINDOW':
        new_detector.signal.array[wind_y0:wind_y0 + wind_y_size, wind_x0:wind_x0 + wind_x_size] += result
    # TODO: add to new_detector.signal.array OR new_detector.image OR charge dataframe ?

    # ng_h2rg.create_hdu(result, 'pyxel/hxrg_read_noise.fits')

    return new_detector


@registry.decorator('signal_transfer', name='nghxrg_u_pink', detector='cmos')
def uncorr_pink_noise(detector: CMOS,
                      u_pink: float = None,
                      window_mode: str = 'FULL',
                      wind_x0: int = 0, wind_y0: int = 0,
                      wind_x_size: int = 0, wind_y_size: int = 0) -> CMOS:
    """TBW.

    :param detector:
    :param u_pink:
    :param window_mode:
    :param wind_x0:
    :param wind_y0:
    :param wind_x_size:
    :param wind_y_size:
    :return:
    """
    new_detector = detector
    geo = t.cast(CMOSGeometry, new_detector.geometry)

    number_of_fits = 1

    ng_h2rg = HXRGNoise(n_out=geo.n_output,
                        nroh=geo.n_row_overhead,
                        nfoh=geo.n_frame_overhead,
                        reverse_scan_direction=geo.reverse_scan_direction,
                        reference_pixel_border_width=geo.reference_pixel_border_width,
                        pca0_file=path.dirname(path.abspath(__file__)) + '/nirspec_pca0.fits',
                        det_size_x=geo.col, det_size_y=geo.row,
                        wind_mode=window_mode,
                        wind_x_size=wind_x_size, wind_y_size=wind_y_size,
                        wind_x0=wind_x0, wind_y0=wind_y0,
                        cube_z=number_of_fits,
                        verbose=True)

    result = ng_h2rg.add_uncorr_pink_noise(u_pink=u_pink)

    result = ng_h2rg.format_result(result)

    if window_mode == 'FULL':
        new_detector.signal.array += result
    elif window_mode == 'WINDOW':
        new_detector.signal.array[wind_y0:wind_y0 + wind_y_size, wind_x0:wind_x0 + wind_x_size] += result
    # TODO: add to new_detector.signal.array OR new_detector.image OR charge dataframe ?

    # ng_h2rg.create_hdu(result, 'pyxel/hxrg_read_noise.fits')

    return new_detector


@registry.decorator('signal_transfer', name='nghxrg_c_pink', detector='cmos')
def corr_pink_noise(detector: CMOS,
                    c_pink: float = None,
                    window_mode: str = 'FULL',
                    wind_x0: int = 0, wind_y0: int = 0,
                    wind_x_size: int = 0, wind_y_size: int = 0) -> CMOS:
    """TBW.

    :param detector:
    :param c_pink:
    :param window_mode:
    :param wind_x0:
    :param wind_y0:
    :param wind_x_size:
    :param wind_y_size:
    :return:
    """
    new_detector = detector
    geo = t.cast(CMOSGeometry, new_detector.geometry)

    number_of_fits = 1

    ng_h2rg = HXRGNoise(n_out=geo.n_output,
                        nroh=geo.n_row_overhead,
                        nfoh=geo.n_frame_overhead,
                        reverse_scan_direction=geo.reverse_scan_direction,
                        reference_pixel_border_width=geo.reference_pixel_border_width,
                        pca0_file=path.dirname(path.abspath(__file__)) + '/nirspec_pca0.fits',
                        det_size_x=geo.col, det_size_y=geo.row,
                        wind_mode=window_mode,
                        wind_x_size=wind_x_size, wind_y_size=wind_y_size,
                        wind_x0=wind_x0, wind_y0=wind_y0,
                        cube_z=number_of_fits,
                        verbose=True)

    result = ng_h2rg.add_corr_pink_noise(c_pink=c_pink)

    result = ng_h2rg.format_result(result)

    if window_mode == 'FULL':
        new_detector.signal.array += result
    elif window_mode == 'WINDOW':
        new_detector.signal.array[wind_y0:wind_y0 + wind_y_size, wind_x0:wind_x0 + wind_x_size] += result
    # TODO: add to new_detector.signal.array OR new_detector.image OR charge dataframe ?

    # ng_h2rg.create_hdu(result, 'pyxel/hxrg_read_noise.fits')

    return new_detector


@registry.decorator('readout_electronics', name='nghxrg_pca_zero', detector='cmos')
def pca_zero_noise(detector: CMOS,
                   pca0_amp: float = None,
                   window_mode: str = 'FULL',
                   wind_x0: int = 0, wind_y0: int = 0,
                   wind_x_size: int = 0, wind_y_size: int = 0) -> CMOS:
    """TBW.

    :param detector:
    :param pca0_amp:
    :param window_mode:
    :param wind_x0:
    :param wind_y0:
    :param wind_x_size:
    :param wind_y_size:
    :return:
    """
    new_detector = detector
    geo = t.cast(CMOSGeometry, new_detector.geometry)
    number_of_fits = 1

    ng_h2rg = HXRGNoise(n_out=geo.n_output,
                        nroh=geo.n_row_overhead,
                        nfoh=geo.n_frame_overhead,
                        reverse_scan_direction=geo.reverse_scan_direction,
                        reference_pixel_border_width=geo.reference_pixel_border_width,
                        pca0_file=path.dirname(path.abspath(__file__)) + '/nirspec_pca0.fits',
                        det_size_x=geo.col, det_size_y=geo.row,
                        wind_mode=window_mode,
                        wind_x_size=wind_x_size, wind_y_size=wind_y_size,
                        wind_x0=wind_x0, wind_y0=wind_y0,
                        cube_z=number_of_fits,
                        verbose=True)

    result = ng_h2rg.add_pca_zero_noise(pca0_amp=pca0_amp)

    result = ng_h2rg.format_result(result)

    if window_mode == 'FULL':
        new_detector.signal.array += result
    elif window_mode == 'WINDOW':
        new_detector.signal.array[wind_y0:wind_y0 + wind_y_size, wind_x0:wind_x0 + wind_x_size] += result
    # TODO: add to new_detector.signal.array OR new_detector.image OR charge dataframe ?

    # ng_h2rg.create_hdu(result, 'pyxel/hxrg_read_noise.fits')

    return new_detector

"""E/B decomposition

Compute E-/B-modes decomposition based on Schneider et al. 2022
(https://arxiv.org/abs/2110.09774)

Author: Axel Guinot

"""

from .schneider2022_nb import (
    _get_pure_EB_modes_parallel,
    _get_pure_EB_modes_serial,
)


def get_pure_EB_modes(
    theta,
    xip,
    xim,
    theta_int,
    xip_int,
    xim_int,
    tmin,
    tmax,
    parallel=False,
    pad_xim=True,
    pad_theta_max_decade=1.0,
    interp_order=5,
    epsabs=1e-10,
    epsrel=1e-10,
):
    """get_pure_EB_modes

    Helper function to enable/disable parallelization.

    Parameters
    ----------
    theta : numpy.ndarray(float64)
        theta in arcmin
    xip : numpy.ndarray(float64)
        xi_plus
    xim : numpy.ndarray(float64)
        xi_minus
    theta_int : numpy.ndarray(float64)
        theta used to cimpute the integrals in arcmin
    xip_int : numpy.ndarray(float64)
        xi_plus used to compute the integrals
    xim_int : numpy.ndarray(float64)
        xi_minus used to compute the integrals
    tmin : float64
        lower bound used for theta in the integrals
    tmax : float64
        upper bound used for theta in the integrals
    parallel : bool
        If True, runs the computation in parallel.
    pad_xim : bool
        If True, pads xim and related arrays to avoid edge effects when
        computing xi_minus E/B-modes.
    pad_theta_max_decade : float64
        Number of decades to pad xim and related arrays to avoid edge effects
        when computing xi_minus E/B-modes.
    interp_order : int
        interpolation order used in the integrals
    epsabs : float64
        absolute error tolerance used in the integrals
    epsrel : float64
        relative error tolerance used in the integrals

    Returns
    -------
    tuple(float64, float64, float64, float64, float64, float64)
        xi_plus_E, xi_minus_E, xi_amb_E, xi_plus_B, xi_minus_B, xi_amb_B
    """

    if parallel:
        return _get_pure_EB_modes_parallel(
            theta,
            xip,
            xim,
            theta_int,
            xip_int,
            xim_int,
            tmin,
            tmax,
            pad_xim=pad_xim,
            pad_theta_max_decade=pad_theta_max_decade,
            interp_order=interp_order,
            epsabs=epsabs,
            epsrel=epsrel,
        )
    else:
        return _get_pure_EB_modes_serial(
            theta,
            xip,
            xim,
            theta_int,
            xip_int,
            xim_int,
            tmin,
            tmax,
            pad_xim=pad_xim,
            pad_theta_max_decade=pad_theta_max_decade,
            interp_order=interp_order,
            epsabs=epsabs,
            epsrel=epsrel,
        )

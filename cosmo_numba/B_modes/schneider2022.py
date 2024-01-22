""" E/B decomposition

Compute E-/B-modes decomposition based on Schneider et al. 2022
(https://arxiv.org/abs/2110.09774)

Author: Axel Guinot

"""
import numba as nb
import numpy as np

from cosmo_numba.math.integrate.quad import interp_quad


@nb.njit(
    nb.float64[:](
        nb.float64,
        nb.float64[:],
        nb.float64,
        nb.float64
    ),
    fastmath=True,
)
def H_p(t, t_int, theta_bar, B):
    """H_p

    See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 46.

    Parameters
    ----------
    t : float64
        theta (fancy)
    t_int : numpy.ndarray(float64)
        theta on which the integral is performed
    theta_bar : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.
    B : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.

    Returns
    -------
    numpy.ndarray(float64)
        H_plus
    """
    return 1/(8*B**3) * \
        (
            4*B**2
            + 3*((t/theta_bar)**2 - 1 - B**2)
            * ((t_int/theta_bar)**2 - 1 - B**2)
        )


@nb.njit(
    nb.float64[:](
        nb.float64,
        nb.float64[:],
        nb.float64,
        nb.float64
    ),
    fastmath=True,
)
def H_m(t, t_int, theta_bar, B):
    """H_m

    See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 47.

    Parameters
    ----------
    t : float64
        theta (fancy)
    t_int : numpy.ndarray(float64)
        theta on which the integral is performed
    theta_bar : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.
    B : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.

    Returns
    -------
    numpy.ndarray(float64)
        H_minus
    """
    return (1-B)**2/(8*B**3) * \
        (
            3*(1-B)**2 * ((1+B)**4-(1+4*B+B**2)*(t/theta_bar)**2) *
            (t_int/theta_bar)**(-2) +
            (3*(1+B)**2*(t/theta_bar)**2-(3+6*B+14*B**2+6*B**3+3*B**4))
        )


@nb.njit(
    nb.float64[:](
        nb.float64,
        nb.float64[:],
        nb.float64,
        nb.float64
    ),
    fastmath=True,
)
def K_p(t, t_int, theta_bar, B):
    """K_p

    See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 51.

    Parameters
    ----------
    t : float64
        theta (fancy)
    t_int : numpy.ndarray(float64)
        theta on which the integral is performed
    theta_bar : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.
    B : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.

    Returns
    -------
    numpy.ndarray(float64)
        K_plus
    """
    return (theta_bar/t)**2 * H_m(t, t_int, B, theta_bar)


@nb.njit(
    nb.float64[:](
        nb.float64,
        nb.float64[:],
        nb.float64,
        nb.float64
    ),
    fastmath=True,
)
def K_m(t, t_int, theta_bar, B):
    """K_m

    See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 54.

    Parameters
    ----------
    t : float64
        theta (fancy)
    t_int : numpy.ndarray(float64)
        theta on which the integral is performed
    theta_bar : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.
    B : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.

    Returns
    -------
    numpy.ndarray(float64)
        K_minus
    """
    return (1-B**2)**2 * theta_bar**4/(t**2*t_int**2) * \
        H_p((1-B**2)*theta_bar**2/t, (1-B**2)*theta_bar**2/t_int, B, theta_bar)


@nb.njit(
    nb.types.Tuple((
        nb.float64,
        nb.float64,
    ))(
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
    ),
    fastmath=True,
)
def get_S(theta_int, xip_int, xim_int, t, tmin, tmax, theta_bar, B):
    """get_S

    See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 44 for
    S_plus and Eq. 45 for S_minus.

    Parameters
    ----------
    theta_int : numpy.ndarray(float64)
        theta used to cimpute the integrals in arcmin
    xip_int : numpy.ndarray(float64)
        xi_plus used to compute the integrals
    xim_int : numpy.ndarray(float64)
        xi_minus used to compute the integrals
    t : float64
        theta (fancy).
    tmin : float64
        lower bound used for theta in the integrals
    tmax : float64
        upper bound used for theta in the integrals
    theta_bar : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.
    B : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.

    Returns
    -------
    tuple(float64, float64)
        S_plus, S_minus
    """

    d_theta_int = np.mean(np.diff(np.log(theta_int)))

    S_p_integrand = 1/theta_bar**2 * theta_int * xip_int \
        * H_p(t, theta_int, B, theta_bar)
    S_p_int = interp_quad(
        np.log(theta_int[0]),
        np.log(theta_int[-1]),
        d_theta_int,
        S_p_integrand,
        tmin,
        tmax,
        k=5,
        padding=True,
        extrap_dist=1,
        log_interp=True,
        epsabs=1e-10, epsrel=1e-10,
    )[0]

    S_m_integrand = 1/theta_int * xim_int * H_m(t, theta_int, B, theta_bar)
    S_m_int = interp_quad(
        np.log(theta_int[0]),
        np.log(theta_int[-1]),
        d_theta_int,
        S_m_integrand,
        tmin,
        tmax,
        k=5,
        padding=True,
        extrap_dist=1,
        log_interp=True,
        epsabs=1e-10, epsrel=1e-10,
    )[0]

    return S_p_int, S_m_int


@nb.njit(
    nb.types.Tuple((
        nb.float64,
        nb.float64,
    ))(
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
    ),
    fastmath=True,
)
def get_V(theta_int, xip_int, xim_int, t, tmin, tmax, theta_bar, B):
    """get_V

    See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 50 for
    V_plus and Eq. 53 for V_minus.

    Parameters
    ----------
    theta_int : numpy.ndarray(float64)
        theta used to cimpute the integrals in arcmin
    xip_int : numpy.ndarray(float64)
        xi_plus used to compute the integrals
    xim_int : numpy.ndarray(float64)
        xi_minus used to compute the integrals
    t : float64
        theta (fancy).
    tmin : float64
        lower bound used for theta in the integrals
    tmax : float64
        upper bound used for theta in the integrals
    theta_bar : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.
    B : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.

    Returns
    -------
    tuple(float64, float64)
        V_plus, V_minus
    """

    d_theta_int = np.mean(np.diff(np.log(theta_int)))

    V_p_integrand = 1/theta_bar**2 * theta_int * xip_int \
        * K_p(t, theta_int, B, theta_bar)
    V_p_int = interp_quad(
        np.log(theta_int[0]),
        np.log(theta_int[-1]),
        d_theta_int,
        V_p_integrand,
        tmin,
        tmax,
        k=5,
        padding=True,
        extrap_dist=1,
        log_interp=True,
        epsabs=1e-10, epsrel=1e-10,
    )[0]

    V_m_integrand = 1/theta_bar**2 * theta_int * xim_int \
        * K_m(t, theta_int, B, theta_bar)
    V_m_int = interp_quad(
        np.log(theta_int[0]),
        np.log(theta_int[-1]),
        d_theta_int,
        V_m_integrand,
        tmin,
        tmax,
        k=5,
        padding=True,
        extrap_dist=1,
        log_interp=True,
        epsabs=1e-10, epsrel=1e-10,
    )[0]

    return V_p_int, V_m_int


@nb.njit(
    nb.types.Tuple((
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
    ))(
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64,
        nb.float64,
    ),
    fastmath=True,
    parallel=True,
)
def get_pure_EB_modes(
    theta, xip, xim,
    theta_int, xip_int, xim_int,
    tmin, tmax
):
    """get_pure_EB_modes

    Computes pure xi E-/B-modes decomposition.
    See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 42-43 for
    xi_plus and Eq. 55-56 for xi_minus.

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

    Returns
    -------
    tuple(float64, float64, float64, float64, float64, float64)
        xi_plus_E, xi_minus_E, xi_amb_E, xi_plus_B, xi_minus_B, xi_amb_B
    """

    n_theta = len(theta)
    xip_E = np.empty(n_theta, dtype=np.float64)
    xip_B = np.empty(n_theta, dtype=np.float64)
    xip_amb = np.empty(n_theta, dtype=np.float64)
    xim_E = np.empty(n_theta, dtype=np.float64)
    xim_B = np.empty(n_theta, dtype=np.float64)
    xim_amb = np.empty(n_theta, dtype=np.float64)

    d_theta_int = np.mean(np.diff(np.log(theta_int)))

    theta_bar = (tmax + tmin)/2
    B = (tmax - tmin)/(tmax + tmin)

    for i in nb.prange(n_theta):
        t = theta[i]

        # xi_p
        m_int = (t < theta_int) & (theta_int < tmax)
        integrand = 1/theta_int[m_int] * xim_int[m_int] \
            * (4 - 12*t**2/theta_int[m_int]**2)

        int_tmp = interp_quad(
            np.log(theta_int[m_int][0]),
            np.log(theta_int[m_int][-1]),
            d_theta_int,
            integrand,
            t,
            tmax,
            k=5,
            padding=True,
            extrap_dist=1,
            log_interp=True,
            epsabs=1e-10, epsrel=1e-10,
        )[0]

        m_int = (theta_int > tmin) & (theta_int < tmax)
        S_p_int, S_m_int = get_S(
            theta_int[m_int], xip_int[m_int], xim_int[m_int],
            t, tmin, tmax,
            theta_bar, B,
        )

        xip_E[i] = 0.5*(xip[i] + xim[i] + int_tmp) - 0.5*(S_p_int + S_m_int)
        xip_B[i] = 0.5*(xip[i] - xim[i] - int_tmp) - 0.5*(S_p_int - S_m_int)
        xip_amb[i] = S_p_int

        # xi_m
        m_int = (t > theta_int) & (theta_int > tmin)
        integrand = theta_int[m_int]/t**2 * xip_int[m_int] \
            * (4 - 12*theta_int[m_int]**2/t**2)

        int_tmp = interp_quad(
            np.log(theta_int[m_int][0]),
            np.log(theta_int[m_int][-1]),
            d_theta_int,
            integrand,
            tmin,
            t,
            k=5,
            padding=True,
            extrap_dist=1,
            log_interp=True,
            epsabs=1e-10, epsrel=1e-10,
        )[0]

        m_int = (theta_int > tmin) & (theta_int < tmax)
        V_p_int, V_m_int = get_V(
            theta_int[m_int], xip_int[m_int], xim_int[m_int],
            t, tmin, tmax,
            theta_bar, B,
        )

        xim_E[i] = 0.5*(xip[i] + xim[i] + int_tmp) - 0.5*(V_p_int + V_m_int)
        xim_B[i] = 0.5*(xip[i] - xim[i] + int_tmp) - 0.5*(V_p_int - V_m_int)
        xim_amb[i] = V_m_int

    return xip_E, xim_E, xip_B, xim_B, xip_amb, xim_amb


@nb.njit(
    nb.types.Tuple((
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
    ))(
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64,
        nb.float64,
    ),
    fastmath=True,
    parallel=True,
)
def get_CNPT_EB_modes(
    theta, xip, xim,
    theta_int, xip_int, xim_int,
    tmin, tmax
):
    """get_pure_EB_modes

    Computes the CNPT xi E-/B-modes decomposition.
    See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 59.

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

    Returns
    -------
    tuple(float64, float64, float64, float64)
        xi_plus_E, xi_minus_E, xi_plus_B, xi_minus_B
    """

    n_theta = len(theta)
    xip_E = np.empty(n_theta, dtype=np.float64)
    xip_B = np.empty(n_theta, dtype=np.float64)
    xim_E = np.empty(n_theta, dtype=np.float64)
    xim_B = np.empty(n_theta, dtype=np.float64)

    d_theta_int = np.mean(np.diff(np.log(theta_int)))

    for i in nb.prange(n_theta):
        t = theta[i]

        # xi_p
        m_int = (t < theta_int) & (theta_int < tmax)
        integrand = 1/theta_int[m_int] * xim_int[m_int] \
            * (4 - 12*t**2/theta_int[m_int]**2)

        int_tmp = interp_quad(
            np.log(theta_int[m_int][0]),
            np.log(theta_int[m_int][-1]),
            d_theta_int,
            integrand,
            t,
            tmax,
            k=5,
            padding=True,
            extrap_dist=1,
            log_interp=True,
            epsabs=1e-10, epsrel=1e-10,
        )[0]

        xip_E[i] = 0.5*(xip[i] + xim[i] + int_tmp)
        xip_B[i] = 0.5*(xip[i] - xim[i] - int_tmp)

        # xi_m
        m_int = (t > theta_int) & (theta_int > tmin)
        integrand = theta_int[m_int]/t**2 * xip_int[m_int] \
            * (4 - 12*theta_int[m_int]**2/t**2)

        int_tmp = interp_quad(
            np.log(theta_int[m_int][0]),
            np.log(theta_int[m_int][-1]),
            d_theta_int,
            integrand,
            tmin,
            t,
            k=5,
            padding=True,
            extrap_dist=1,
            log_interp=True,
            epsabs=1e-10, epsrel=1e-10,
        )[0]

        xim_E[i] = 0.5*(xip[i] + xim[i] + int_tmp)
        xim_B[i] = 0.5*(xip[i] - xim[i] + int_tmp)

    return xip_E, xim_E, xip_B, xim_B

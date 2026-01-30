"""COSEBIs numba

Here you find all the computation necessary for the COSEBIs performed in using
Numba.

Author: Axel Guinot

"""

import os

import numba as nb
import numpy as np

from ..math.integrate.fftlog import fht
from ..math.interpolate.interpolate_1D import AkimaInterp1D
from ..math.utils import extend_log_grid, pad_1D

from .cosebis_nb_cfunc import tm_log_nb

from ..math.integrate.simpson import simpson

# This allows having coverage
IN_COVERAGE = False
if (
    os.environ.get("TESTING_COSEBIS", "0") == "1"
    and os.environ.get("COVERAGE_MODE", "0") == "1"
):
    nb.config.DISABLE_JIT = 1
    IN_COVERAGE = True
    from ..math.interpolate.interpolate_1D import (
        _AkimaInterp1D as AkimaInterp1D,
    )


@nb.njit
def tp_log_nb(z, roots, norm):
    """tp_log

    Compute `tp_log(z)` for a specific mode based on the provided roots and
    norm.
    See Schneider et al. 2010 (https://arxiv.org/abs/1002.2136) Eq. 36.

    Parameters
    ----------
    z : numpy.ndarray(float64)
        z representing ln(theta/theta_min).
    roots : numpy.ndarray(float64)
        Pre-computed roots for a given mode.
    norm : float64
        Pre-computed normalization for a given mode.

    Returns
    -------
    numpy.ndarray(float64)
        tp_log(z) for a given mode
    """
    n_z = len(z)
    n_roots = len(roots)
    out = np.empty(n_z, dtype=np.float64)
    for i in nb.prange(n_z):
        tmp = 1
        for j in nb.prange(n_roots):
            tmp *= z[i] - roots[j]
        out[i] = norm * tmp
    return out


@nb.njit(
    parallel=True,
)
def tp_n_log(z, roots_n, norms_n):
    """tp_n_log

    Compute `tp_log(n, z)` for all modes `n` based on the provided roots and
    norms.

    Parameters
    ----------
    z : numpy.ndarray(float64)
        z representing ln(theta/theta_min).
    roots_n : numba.types.ListType(numpy.ndarray(float64))
        Pre-computed roots for all modes.
    norms_n : numpy.ndarray(float64)
        Pre-computed normalization for all modes.

    Returns
    -------
    numpy.ndarray(float64)
        tp_log(n, z)
    """

    Nmodes = len(roots_n)
    n_z = len(z)
    arr_tp_n_log = np.empty((Nmodes, n_z), dtype=np.float64)
    for n in nb.prange(Nmodes):
        n = np.int64(n)
        arr_tp_n_log[n, :] = tp_log_nb(z, roots_n[n], norms_n[n])

    return arr_tp_n_log


@nb.njit(
    parallel=True,
)
def tm_n_log(z, roots_n, norms_n):
    """tm_n_log

    Compute `tm_log(n, z)` for all modes `n` based on the provided roots and
    norms.

    Parameters
    ----------
    z : numpy.ndarray(float64)
        z representing ln(theta/theta_min).
    roots_n : numba.types.ListType(numpy.ndarray(float64))
        Pre-computed roots for all modes.
    norms_n : numpy.ndarray(float64)
        Pre-computed normalization for all modes.

    Returns
    -------
    numpy.ndarray(float64)
        tm_log(n, z)
    """

    Nmodes = len(roots_n)
    n_z = len(z)
    arr_tm_n_log = np.empty((Nmodes, n_z), dtype=np.float64)
    for n in nb.prange(Nmodes):
        n = np.int64(n)
        arr_tm_n_log[n, :] = tm_log_nb(z, roots_n[n], norms_n[n])

    return arr_tm_n_log


@nb.njit(
    parallel=True,
)
def wn_log(
    ell,
    theta_rad,
    Tp_log,
    q,
    pad_low_decades,
    pad_high_decades,
):
    """wn_log

    Compute `Wn_log(n, z)` for all modes `n` based on the provided
    `Tp_log(n, theta)`.
    See Schneider et al. 2010 (https://arxiv.org/abs/1002.2136) Eq. 6.

    NOTE: Running this in parallel with numba leads to errors.
    I don't know why..

    Parameters
    ----------
    ell : numpy.ndarray(float64)
        ell values where to compute Wn_log.
    theta_rad : numpy.ndarray(float64)
        Theta in radians
    Tp_log : numpy.ndarray((float64, float64))
        Pre-computed Tp_log for all modes.
    q : float64
        FFTLog parameter q.
    pad_low_decades : float64
        Number of decades to pad at low theta.
    pad_high_decades : float64
        Number of decades to pad at high theta.

    Returns
    -------
    numpy.ndarray((float64, float64))
        Wn_log(n, ell)
    """

    theta_rad_ext, i_low, i_high = extend_log_grid(
        theta_rad, pad_low_decades, pad_high_decades
    )

    nbins = int(len(theta_rad_ext))
    N_mode = Tp_log.shape[0]

    Wn_log = np.empty((N_mode, len(ell)), dtype=np.float64)
    for n in nb.prange(N_mode):
        # We apply padding to Tp_log to avoid edge effects in the FFTLog.
        # Particularly for small theta.
        Tp_log_ext = pad_1D(nbins, i_low, i_high, Tp_log[n])
        ell_, Wn_log_tmp = fht(
            nbins, theta_rad_ext, Tp_log_ext, 2.0, 0.0, q, 1, 1
        )
        interp = AkimaInterp1D(
            ell_,
            Wn_log_tmp * 2.0 * np.pi,
        )
        Wn_log[n] = interp.eval(ell)

    return Wn_log


@nb.njit
def get_xipm_cosebis_serial(
    theta_rad,
    xip,
    xim,
    Tp_log,
    Tm_log,
    N_mode,
):
    """get_xipm_cosebis

    Compute the COSEBIs from the 2PCF xi_plus and xi_minus.
    See Schneider et al. 2010 (https://arxiv.org/abs/1002.2136) Eq. 1.

    Parameters
    ----------
    theta_rad : numpy.ndarray(float64)
        theta in radian
    xip : numpy.ndarray(float64)
        2PCF xi_plus
    xim : numpy.ndarray(float64)
        2PCF xi_minus
    Tp_log : numpy.ndarray(float64)
        Window function Tp_log
    Tm_log : numpy.ndarray(float64)
        Window function Tm_log
    N_mode : int64
        Number of mode to compute

    Returns
    -------
    tuple(numpy.ndarray(float64), numpy.ndarray(float64))
        COSEBIs E- and B-modes.
    """

    C_E = np.empty(N_mode, dtype=np.float64)
    C_B = np.empty(N_mode, dtype=np.float64)
    for n in nb.prange(N_mode):
        integrand_E = theta_rad * (Tp_log[n] * xip + Tm_log[n] * xim) * 0.5
        integrand_B = theta_rad * (Tp_log[n] * xip - Tm_log[n] * xim) * 0.5

        C_E[n] = simpson(integrand_E, theta_rad)
        C_B[n] = simpson(integrand_B, theta_rad)

    return C_E, C_B


if IN_COVERAGE:  # pragma: no cover
    get_xipm_cosebis_parallel = get_xipm_cosebis_serial
else:  # pragma: no cover
    get_xipm_cosebis_parallel = nb.njit(
        parallel=True,
    )(get_xipm_cosebis_serial.py_func)


@nb.njit
def get_Cell_cosebis_serial(ell, Cell_E, Cell_B, Wn_log, N_mode):
    """get_xipm_cosebis

    Compute the COSEBIs from the Cell E- and B-modes.
    See Schneider et al. 2010 (https://arxiv.org/abs/1002.2136) Eq. 5.

    Parameters
    ----------
    ell : numpy.ndarray(float64)
        ell
    Cell_E : numpy.ndarray(float64)
        Power spectrum E-modes
    Cell_B : numpy.ndarray(float64)
        Power spectrum B-modes
    Wn_log : numba.types.ListType(AkimaInterp1D.class_type.instance_type)
        Window function Wn_log
    N_mode : int64
        Number of mode to compute

    Returns
    -------
    tuple(numpy.ndarray(float64), numpy.ndarray(float64))
        COSEBIs E- and B-modes.
    """

    C_E = np.empty(N_mode, dtype=np.float64)
    C_B = np.empty(N_mode, dtype=np.float64)
    for n in nb.prange(N_mode):
        integrand_E = ell * Cell_E * Wn_log[n] / np.pi * 0.5
        integrand_B = ell * Cell_B * Wn_log[n] / np.pi * 0.5
        C_E[n] = simpson(integrand_E, ell)
        C_B[n] = simpson(integrand_B, ell)

    return C_E, C_B


if IN_COVERAGE:  # pragma: no cover
    get_Cell_cosebis_parallel = get_Cell_cosebis_serial
else:  # pragma: no cover
    get_Cell_cosebis_parallel = nb.njit(
        parallel=True,
    )(get_Cell_cosebis_serial.py_func)


@nb.njit(
    parallel=True,
)
def get_cosebis_cov_from_xipm_cov(theta_rad, cov_xipm, Tp_log, Tm_log, N_mode):
    """get_cosebis_cov_from_xipm_cov

    Compute the covariace of the COSEBIs from the shear-shear covariance of
    xi_plus and xi_minus.
    See Schneider et al. 2010 (https://arxiv.org/abs/1002.2136) Eq. 9.

    Parameters
    ----------
    theta_rad : numpy.ndarray(float64)
        theta in radian
    cov_xipm : numpy.ndarray(float64)
        shear-shear covariance
    Tp_log : numpy.ndarray(float64)
        Window function Tp_log
    Tm_log : numpy.ndarray(float64)
        Window function Tm_log
    N_mode : int64
        Number of mode to compute

    Returns
    -------
    numpy.ndarray(float64)
        COSEBIs covariance
    """

    n_theta = len(theta_rad)
    n_bins = np.int64(cov_xipm.shape[0] / 2)

    cov_En = np.zeros((N_mode, N_mode), dtype=np.float64)
    cov_Bn = np.zeros((N_mode, N_mode), dtype=np.float64)
    cov_EBn = np.zeros((N_mode, N_mode), dtype=np.float64)
    for m in nb.prange(N_mode):
        for n in nb.prange(N_mode):
            integ_E_tmp = np.empty(len(theta_rad), dtype=np.float64)
            integ_B_tmp = np.empty(len(theta_rad), dtype=np.float64)
            for i in nb.prange(n_theta):
                # E-modes
                integ_E_tmp2 = theta_rad * (
                    Tp_log[m][i] * Tp_log[n] * cov_xipm[i, :n_bins]
                    + Tp_log[m][i] * Tm_log[n] * cov_xipm[i, n_bins:]
                    + Tm_log[m][i] * Tp_log[n] * cov_xipm[n_bins + i, :n_bins]
                    + Tm_log[m][i] * Tm_log[n] * cov_xipm[n_bins + i, n_bins:]
                )
                integ_E_tmp[i] = simpson(integ_E_tmp2, theta_rad)

                # B-modes
                integ_B_tmp2 = theta_rad * (
                    Tp_log[m][i] * Tp_log[n] * cov_xipm[i, :n_bins]
                    - Tp_log[m][i] * Tm_log[n] * cov_xipm[i, n_bins:]
                    - Tm_log[m][i] * Tp_log[n] * cov_xipm[n_bins + i, :n_bins]
                    + Tm_log[m][i] * Tm_log[n] * cov_xipm[n_bins + i, n_bins:]
                )
                integ_B_tmp[i] = simpson(integ_B_tmp2, theta_rad)
            cov_En[m, n] = (
                1 / 4.0 * simpson(integ_E_tmp * theta_rad, theta_rad)
            )
            cov_Bn[m, n] = (
                1 / 4.0 * simpson(integ_B_tmp * theta_rad, theta_rad)
            )
            cov_EBn[m, n] = 0.0

    cov_EB_full = np.empty((N_mode * 2, N_mode * 2), dtype=np.float64)
    cov_EB_full[:N_mode, :N_mode] = cov_En
    cov_EB_full[N_mode:, N_mode:] = cov_Bn
    cov_EB_full[:N_mode, N_mode:] = cov_EBn
    cov_EB_full[N_mode:, :N_mode] = cov_EBn

    return cov_EB_full

""" COSEBIs numba

Here you find all the computation necessary for the COSEBIs performed in using
Numba.

Author: Axel Guinot

"""
import numba as nb
import numpy as np

from NumbaQuadpack import dqags

from cosmo_numba.math.integrate.simpson import simpson
from cosmo_numba.fftlog import fht
from cosmo_numba.math.interpolate.interpolate_1D import AkimaInterp1D


@nb.njit(
    nb.float64[:](nb.float64[:], nb.float64[:], nb.float64),
    fastmath=True,
)
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
    out = np.empty(n_z, dtype=np.float64)
    for i in range(n_z):
        tmp = 1
        for root in roots:
            tmp *= z[i] - root
        out[i] = norm*tmp
    return out


@nb.njit(
    nb.float64[:, :](
        nb.float64[:],
        nb.types.ListType(
            nb.types.Array(nb.float64, 1, 'C', False, aligned=True)
        ),
        nb.float64[:]
    ),
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
    norm : numpy.ndarray(float64)
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
        arr_tp_n_log[n, :] = tp_log_nb(z, roots_n[n], norms_n[n])

    return arr_tp_n_log


# This function does the same thing as the one above but for scalar value of z.
# It is also written to be easier to call during the integration for the
# computation of tm_log
@nb.njit(
    nb.float64(nb.float64, nb.float64[:]),
    fastmath=True,
)
def _tp_log_nb(z, data):
    """tp_log

    Compute `tp_log(z)` for a specific mode based on the provided roots and
    norm.
    See Schneider et al. 2010 (https://arxiv.org/abs/1002.2136) Eq. 36.

    Parameters
    ----------
    z : float64
        z representing ln(theta/theta_min).
    data: numpy.ndarray(float64)
        data[0] = n_roots = len(roots)
        data[1:n_roots+1] = roots
        data[n_roots+1] = norm

    Returns
    -------
    float64
        tp_log(z) for a given mode
    """

    n_roots = int(data[0])
    roots = np.empty(n_roots, dtype=np.float64)
    for i in range(1, n_roots+1):
        roots[i-1] = data[i]
    norm = data[n_roots+1]

    out = 1
    for root in roots:
        out *= z - root
    out *= norm
    return out


##
spec_integ_tm_log = nb.float64(nb.float64, nb.types.CPointer(nb.float64))


@nb.cfunc(
    spec_integ_tm_log,
)
@nb.njit(
    spec_integ_tm_log,
    fastmath=True,
)
def _integrand_tm_log(y, data_integ_):
    """integrand tm_log

    Compute the integrand for the integral of Eq. 37 (first line).

    Parameters
    ----------
    y : float64
        integration variable
    data_integ : numba.carray(float64)
        Contains the necessary information to compute tp_log
        (see `_tp_log_nb`).

    Returns
    -------
    float64
        Integrand.
    """

    len_data_tp_log = int(data_integ_[0])+3
    data_integ = nb.carray(data_integ_, (len_data_tp_log,), np.float64)
    tp_log_y = _tp_log_nb(y, data_integ[:-1])

    z = np.float64(data_integ[-1])
    fac = np.exp(2.*(y - z)) - 3.*np.exp(4.*(y - z))

    return tp_log_y*fac


# We need to do this because the attribute `.address` is not accessible from
# within a jitted function.
INTEG_TM_LOG_ADDRESS = nb.experimental.function_type._get_wrapper_address(
    _integrand_tm_log,
    spec_integ_tm_log,
)


@nb.njit(
    nb.float64[:](nb.float64[:], nb.float64[:], nb.float64),
    fastmath=True,
)
def tm_log_nb(z, roots, norm):
    """tm_log

    Compute `tm_log(z)` for a specific mode based on the provided roots and
    norm.
    See Schneider et al. 2010 (https://arxiv.org/abs/1002.2136) Eq. 37 (first
    line).

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
        tm_log(z) for a given mode
    """

    # build data for integration
    n_roots = len(roots)
    data = np.empty(n_roots+2, dtype=np.float64)
    data_integ = np.empty(n_roots+3, dtype=np.float64)
    data[0] = n_roots
    data[1:n_roots+1] = roots
    data[n_roots+1] = norm
    data_integ[:-1] = data

    n_z = len(z)
    tm_log = np.empty(n_z, dtype=np.float64)
    for i in range(n_z):
        data_integ[-1] = z[i]
        tp_z = _tp_log_nb(z[i], data)
        res_int = dqags(
            INTEG_TM_LOG_ADDRESS,
            0.,
            z[i],
            data=data_integ
        )
        tm_log[i] = tp_z + 4. * res_int[0]

    return tm_log


@nb.njit(
    nb.float64[:, :](
        nb.float64[:],
        nb.types.ListType(
            nb.types.Array(nb.float64, 1, 'C', False, aligned=True)
        ),
        nb.float64[:]
    ),
    parallel=True,
)
def tm_n_log(z, roots_n, norms_n):
    """tp_n_log

    Compute `tm_log(n, z)` for all modes `n` based on the provided roots and
    norms.

    Parameters
    ----------
    z : numpy.ndarray(float64)
        z representing ln(theta/theta_min).
    roots_n : numba.types.ListType(numpy.ndarray(float64))
        Pre-computed roots for all modes.
    norm : numpy.ndarray(float64)
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
        arr_tm_n_log[n, :] = tm_log_nb(z, roots_n[n], norms_n[n])

    return arr_tm_n_log


@nb.njit(
    nb.types.ListType(
        AkimaInterp1D.class_type.instance_type,
    )(
        nb.float64[:],
        nb.float64[:, :],
        nb.float64,
    ),
    fastmath=True,
    parallel=False,
)
def wn_log(theta_rad, Tp_log, q):
    """wn_log

    Compute `Wn_log(n, z)` for all modes `n` based on the provided
    `Tp_log(n, theta)`.
    See Schneider et al. 2010 (https://arxiv.org/abs/1002.2136) Eq. 6.

    NOTE: Running this in parallel with numba leads to errors.
    I don't know why..

    Parameters
    ----------
    theta_rad : numpy.ndarray(float64)
        Theta in radians
    Tp_log : numpy.ndarray((float64, float64))
        Pre-computed Tp_log for all modes.

    Returns
    -------
    numba.types.ListType(AkimaInterp1D.class_type.instance_type)
        Wn_log(n, l)
    """

    nbins = int(len(theta_rad))
    N_mode = Tp_log.shape[0]

    Wn_log = nb.typed.List()
    for n in nb.prange(N_mode):
        ell_, Wn_log_tmp = fht(
            nbins,
            theta_rad,
            Tp_log[n],
            2.,
            0.,
            q,
            1,
            1
        )
        Wn_log.append(
            AkimaInterp1D(
                ell_, Wn_log_tmp*2.*np.pi,
            )
        )

    return Wn_log


@nb.njit(
    nb.types.Tuple((nb.float64[:], nb.float64[:]))(
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.int64,
    ),
    fastmath=True,
    parallel=True,
)
def get_xipm_cosebis(theta_rad, dtheta_rad, xip, xim, Tp_log, Tm_log, N_mode):
    """get_xipm_cosebis

    Compute the COSEBIs from the 2PCF xi_plus and xi_minus.
    See Schneider et al. 2010 (https://arxiv.org/abs/1002.2136) Eq. 1.

    Parameters
    ----------
    theta_rad : numpy.ndarray(float64)
        theta in radian
    dtheta_rad : numpy.ndarray(float64)
        bin size for theta in radian
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
        integrand_E = theta_rad * (Tp_log[n]*xip + Tm_log[n]*xim) * 0.5
        integrand_B = theta_rad * (Tp_log[n]*xip - Tm_log[n]*xim) * 0.5

        C_E[n] = np.sum(integrand_E * dtheta_rad)
        C_B[n] = np.sum(integrand_B * dtheta_rad)

    return C_E, C_B


@nb.njit(
    nb.types.Tuple((nb.float64[:], nb.float64[:]))(
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.types.ListType(AkimaInterp1D.class_type.instance_type),
        nb.int64,
    ),
    fastmath=True,
    parallel=True,
)
def get_Cell_cosebis(ell, Cell_E, Cell_B, Wn_log, N_mode):
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
        integrand_E = \
            ell * Cell_E * Wn_log[n].eval(ell) / np.pi * 0.5
        integrand_B = \
            ell * Cell_B * Wn_log[n].eval(ell) / np.pi * 0.5
        C_E[n] = simpson(integrand_E, ell)
        C_B[n] = simpson(integrand_B, ell)

    return C_E, C_B


@nb.njit(
    nb.float64[:, :](
        nb.float64[:],
        nb.float64[:],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.int64,
    ),
    fastmath=True,
    parallel=True,
)
def get_cosebis_cov_from_xipm_cov(
    theta_rad,
    dtheta_rad,
    cov_xipm,
    Tp_log,
    Tm_log,
    N_mode
):
    """get_cosebis_cov_from_xipm_cov

    Compute the covariace of the COSEBIs from the shear-shear covariance of
    xi_plus and xi_minus.
    See Schneider et al. 2010 (https://arxiv.org/abs/1002.2136) Eq. 9.

    Parameters
    ----------
    theta_rad : numpy.ndarray(float64)
        theta in radian
    dtheta_rad : numpy.ndarray(float64)
        bin size for theta in radian
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
    n_bins = np.int64(cov_xipm.shape[0]/2)

    cov_En = np.zeros((N_mode, N_mode), dtype=np.float64)
    cov_Bn = np.zeros((N_mode, N_mode), dtype=np.float64)
    cov_EBn = np.zeros((N_mode, N_mode), dtype=np.float64)
    for m in nb.prange(N_mode):
        for n in nb.prange(N_mode):
            integ_E_tmp = np.empty(len(theta_rad), dtype=np.float64)
            integ_B_tmp = np.empty(len(theta_rad), dtype=np.float64)
            for i in nb.prange(n_theta):
                # E-modes
                integ_E_tmp2 = theta_rad*(
                    Tp_log[m][i]*Tp_log[n]*cov_xipm[i, :n_bins]
                    + Tp_log[m][i]*Tm_log[n]*cov_xipm[i, n_bins:]
                    + Tm_log[m][i]*Tp_log[n]*cov_xipm[n_bins+i, :n_bins]
                    + Tm_log[m][i]*Tm_log[n]*cov_xipm[n_bins+i, n_bins:]
                )
                integ_E_tmp[i] = np.sum(integ_E_tmp2 * dtheta_rad)

                # B-modes
                integ_B_tmp2 = theta_rad*(
                    Tp_log[m][i]*Tp_log[n]*cov_xipm[i, :n_bins]
                    - Tp_log[m][i]*Tm_log[n]*cov_xipm[i, n_bins:]
                    - Tm_log[m][i]*Tp_log[n]*cov_xipm[n_bins+i, :n_bins]
                    + Tm_log[m][i]*Tm_log[n]*cov_xipm[n_bins+i, n_bins:]
                )
                integ_B_tmp[i] = np.sum(integ_B_tmp2 * dtheta_rad)
            cov_En[m, n] = 1/4. * np.sum(integ_E_tmp*theta_rad * dtheta_rad)
            cov_Bn[m, n] = 1/4. * np.sum(integ_B_tmp*theta_rad * dtheta_rad)
            cov_EBn[m, n] = 0.

    cov_EB_full = np.empty((N_mode*2, N_mode*2), dtype=np.float64)
    cov_EB_full[:N_mode, :N_mode] = cov_En
    cov_EB_full[N_mode:, N_mode:] = cov_Bn
    cov_EB_full[:N_mode, N_mode:] = cov_EBn
    cov_EB_full[N_mode:, :N_mode] = cov_EBn

    return cov_EB_full

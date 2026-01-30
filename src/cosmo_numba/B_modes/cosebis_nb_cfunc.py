import numpy as np
import numba as nb

from NumbaQuadpack import dqags


# This function does the same thing as the one above but for scalar value of z.
# It is also written to be easier to call during the integration for the
# computation of tm_log
@nb.njit(
    nb.float64(nb.float64, nb.float64[:]),
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
    for i in nb.prange(1, n_roots + 1):
        roots[i - 1] = data[i]
    norm = data[n_roots + 1]

    out = 1
    for root in roots:
        out *= z - root
    out *= norm
    return out


spec_integ_tm_log = nb.float64(nb.float64, nb.types.CPointer(nb.float64))


@nb.cfunc(
    spec_integ_tm_log,
)
@nb.njit(
    spec_integ_tm_log,
)
def _integrand_tm_log(y, data_integ_):
    """integrand tm_log

    Compute the integrand for the integral of Eq. 37 (first line).

    Parameters
    ----------
    y : float64
        integration variable
    data_integ_ : numba.carray(float64)
        Contains the necessary information to compute tp_log
        (see `_tp_log_nb`).

    Returns
    -------
    float64
        Integrand.
    """

    len_data_tp_log = int(data_integ_[0]) + 3
    data_integ = nb.carray(data_integ_, (len_data_tp_log,), np.float64)
    tp_log_y = _tp_log_nb(y, data_integ[:-1])

    z = np.float64(data_integ[-1])
    fac = np.exp(2.0 * (y - z)) - 3.0 * np.exp(4.0 * (y - z))

    return tp_log_y * fac


# We need to do this because the attribute `.address` is not accessible from
# within a jitted function.
INTEG_TM_LOG_ADDRESS = nb.experimental.function_type._get_wrapper_address(
    _integrand_tm_log,
    spec_integ_tm_log,
)


@nb.njit(
    nb.float64[:](nb.float64[:], nb.float64[:], nb.float64),
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
    data = np.empty(n_roots + 2, dtype=np.float64)
    data_integ = np.empty(n_roots + 3, dtype=np.float64)
    data[0] = n_roots
    data[1 : n_roots + 1] = roots
    data[n_roots + 1] = norm
    data_integ[:-1] = data

    n_z = len(z)
    tm_log = np.empty(n_z, dtype=np.float64)
    for i in nb.prange(n_z):
        data_integ[-1] = z[i]
        tp_z = _tp_log_nb(z[i], data)
        res_int = dqags(INTEG_TM_LOG_ADDRESS, 0.0, z[i], data=data_integ)
        tm_log[i] = tp_z + 4.0 * res_int[0]

    return tm_log

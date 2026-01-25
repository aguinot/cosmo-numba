import numpy as np
import numba as nb


@nb.njit(
    nb.complex128(nb.complex128),
    fastmath=True,
)
def lngamma(z):
    """
    Numerical Recipes 6.1

    From: https://stackoverflow.com/questions/55048299/why-is-this-log-gamma-numba-function-slower-than-scipy-for-large-arrays-but-fas
    """

    coefs = np.array(
        [
            57.1562356658629235,
            -59.5979603554754912,
            14.1360979747417471,
            -0.491913816097620199,
            0.339946499848118887e-4,
            0.465236289270485756e-4,
            -0.983744753048795646e-4,
            0.158088703224912494e-3,
            -0.210264441724104883e-3,
            0.217439618115212643e-3,
            -0.164318106536763890e-3,
            0.844182239838527433e-4,
            -0.261908384015814087e-4,
            0.368991826595316234e-5,
        ]
    )

    y = z
    tmp = z + 5.24218750000000000
    tmp = (z + 0.5) * np.log(tmp) - tmp
    ser = 0.999999999999997092

    n = coefs.shape[0]
    for j in range(n):
        y = y + 1.0
        ser = ser + coefs[j] / y

    out = tmp + np.log(2.5066282746310005 * ser / z)
    return out


@nb.njit(
    nb.types.Tuple((nb.float64[:], nb.int64, nb.int64))(
        nb.float64[:],
        nb.float64,
        nb.float64,
    ),
)
def extend_log_grid(theta, pad_low_decades=2.0, pad_high_decades=2.0):
    """
    Extend a log-spaced theta grid using its native log spacing.

    Parameters
    ----------
    theta : array_like
        Original log-spaced theta grid (monotonic increasing).
    pad_low_decades, pad_high_decades : float
        Padding in decades below and above the original grid.

    Returns
    -------
    theta_ext : ndarray
        Extended log-spaced theta grid.
    i0 : int
        Starting index of original theta in theta_ext.
    """
    theta = np.asarray(theta)

    # Get log-spacing
    dln = np.median(np.diff(np.log(theta)))

    # Number of points to pad
    n_low = int(np.round(pad_low_decades * np.log(10.0) / dln))
    n_high = int(np.round(pad_high_decades * np.log(10.0) / dln))

    # Build lower extension (strictly below theta[0])
    ln_theta0 = np.log(theta[0])
    ln_low = ln_theta0 - dln * np.arange(n_low, 0, -1)
    theta_low = np.exp(ln_low)

    # Build upper extension (strictly above theta[-1])
    ln_theta1 = np.log(theta[-1])
    ln_high = ln_theta1 + dln * np.arange(1, n_high + 1)
    theta_high = np.exp(ln_high)

    theta_ext = np.empty(n_low + theta.size + n_high, dtype=theta.dtype)
    theta_ext[:n_low] = theta_low
    theta_ext[n_low : n_low + theta.size] = theta
    theta_ext[n_low + theta.size :] = theta_high

    return theta_ext, n_low, n_low + theta.size


@nb.njit(
    nb.float64[:](
        nb.int64,
        nb.int64,
        nb.int64,
        nb.float64[:],
    ),
)
def pad_1D(N_theta_ext, i_low, i_high, f_theta):
    """
    Extend f(theta) into an extended theta grid.
    """

    f_theta_ext = np.zeros(N_theta_ext)
    f_theta_ext[i_low:i_high] = f_theta
    return f_theta_ext


@nb.njit(
    nb.float64[:](nb.float64[:]),
    fastmath=True,
)
def numbadiff(x):
    """
    Compute the difference between consecutive elements of an array. Numba
    implementation of np.diff.

    Parameters
    ----------
    x : float64[:]
        input array

    Returns
    -------
    float64[:]
        differences between consecutive elements with size len(x)-1.
    """
    return x[1:] - x[:-1]

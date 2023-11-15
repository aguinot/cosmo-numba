import numpy as np
import numba as nb


@nb.njit(
    [
        nb.complex128[:](
            nb.float64[:]
        ),
        nb.complex128[:](
            nb.complex128[:]
        )
    ],
)
def compute_fft(x):
    y = np.zeros(len(x), dtype=np.complex128)
    with nb.objmode(y='complex128[:]'):
        y = np.fft.fft(x)
    return y


@nb.njit(
    [
        nb.complex128[:](
            nb.float64[:]
        ),
        nb.complex128[:](
            nb.complex128[:]
        )
    ],
)
def compute_ifft(x):
    y = np.zeros(len(x), dtype=np.complex128)
    with nb.objmode(y='complex128[:]'):
        y = np.fft.ifft(x)
    return y


@nb.njit(
    nb.complex128(nb.complex128),
    fastmath=True,
)
def lngamma(z):
    """
    Numerical Recipes 6.1

    From: https://stackoverflow.com/questions/55048299/why-is-this-log-gamma-numba-function-slower-than-scipy-for-large-arrays-but-fas  # noqa
    """

    coefs = np.array([
        57.1562356658629235, -59.5979603554754912,
        14.1360979747417471, -0.491913816097620199,
        .339946499848118887e-4, .465236289270485756e-4,
        -.983744753048795646e-4, .158088703224912494e-3,
        -.210264441724104883e-3, .217439618115212643e-3,
        -.164318106536763890e-3, .844182239838527433e-4,
        -.261908384015814087e-4, .368991826595316234e-5
    ])

    y = z
    tmp = z + 5.24218750000000000
    tmp = (z + 0.5) * np.log(tmp) - tmp
    ser = 0.999999999999997092

    n = coefs.shape[0]
    for j in range(n):
        y = y + 1.
        ser = ser + coefs[j] / y

    out = tmp + np.log(2.5066282746310005 * ser / z)
    return out

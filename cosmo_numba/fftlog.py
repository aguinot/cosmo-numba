"""FFTLog Numba

First imlplemented by Andrew Hamilton:

http://casa.colorado.edu/~ajsh/FFTLog/

Adapted from C in CCL:
https://github.com/LSSTDESC/CCL/blob/master/src/ccl_fftlog.c

Author: Axel Guinot

"""

import numpy as np
import numba as nb

from .math.utils import lngamma, compute_fft, compute_ifft


@nb.njit(
    nb.types.Tuple((nb.float64, nb.float64))(nb.float64, nb.float64),
    fastmath=True,
)
def lngamma_4(x, y):
    out = lngamma(x + y * 1j)
    return np.real(out), np.imag(out)


@nb.njit(
    nb.complex128(nb.float64, nb.float64),
    fastmath=True,
)
def polar(r, phi):
    res = r * np.cos(phi) + r * np.sin(phi) * 1j
    return res


@nb.njit(
    nb.float64(
        nb.int64,
        nb.int64,
        nb.int64,
        nb.int64,
        nb.float64,
    ),
    fastmath=True,
)
def goodkr(N, mu, q, L, kr):
    xp = (mu + 1 + q) / 2.0
    xm = (mu + 1 - q) / 2.0
    y = np.pi * N / (2.0 * L)
    lnr, argp = lngamma_4(xp, y)
    lnr, argm = lngamma_4(xm, y)
    arg = np.log(2 / kr) * N / L + (argp + argm) / np.pi
    iarg = np.round(arg)
    if arg != iarg:
        kr *= np.exp((arg - iarg) * L / N)
    return kr


@nb.njit(
    nb.void(
        nb.int64,
        nb.int64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.complex128[:],
    ),
    fastmath=True,
)
def compute_u_coeff(N, mu, q, L, kcrc, u):
    y = np.pi / L
    k0r0 = kcrc * np.exp(-L)
    t = -2.0 * y * np.log(k0r0 / 2.0)

    if q == 0:
        x = (mu + 1) / 2.0
        for m in range(0, int(N / 2) + 1):
            lnr, phi = lngamma_4(x, m * y)
            u[m] = polar(1.0, m * t + 2 * phi)
    else:
        xp = (mu + 1 + q) / 2.0
        xm = (mu + 1 - q) / 2.0
        for m in range(0, int(N / 2) + 1):
            lnrp, phip = lngamma_4(xp, m * y)
            lnrm, phim = lngamma_4(xm, -m * y)
            u[m] = polar(
                np.exp(q * np.log(2.0) + lnrp - lnrm), m * t + phip - phim
            )

    for m in range(int(N / 2) + 1, N):
        u[m] = np.conj(u[N - m])
    if np.mod(N, 2) == 0:
        u[int(N / 2)] = np.real(u[int(N / 2)]) * 0.0 * 1j


@nb.njit(
    nb.types.Tuple(
        (
            nb.types.Array(nb.float64, 1, "C", False, aligned=True),
            nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        )
    )(
        nb.int64,
        nb.float64[:],
        nb.float64[:],
        nb.int64,
        nb.int64,
        nb.float64,
        nb.int64,
        nb.int64,
    ),
    fastmath=True,
)
def fht(N, k, pk, dim, mu, q, kcrc, noring):
    L = np.log(k[N - 1] / k[0]) * N / (N - 1.0)
    if noring:
        kcrc = goodkr(N, mu, q, L, kcrc)

    u = np.empty(N, dtype=np.complex128)
    compute_u_coeff(N, mu, q, L, kcrc, u)

    prefac_pk = np.power(k, dim / 2.0 - q)
    k0r0 = kcrc * np.exp(-L)
    r = np.zeros(N)
    r[0] = k0r0 / k[0]
    for n in range(1, N):
        r[n] = r[0] * np.exp(n * L / N)

    one_over_2pi_dhalf = pow(2 * np.pi, -dim / 2)
    prefac_xi = one_over_2pi_dhalf * np.power(r, -dim / 2 - q)

    a = prefac_pk * pk
    b = compute_fft(a)
    b *= u
    b = compute_ifft(b)

    for n in range(0, int(N / 2)):
        tmp = b[n]
        b[n] = b[N - n - 1]
        b[N - n - 1] = tmp
    xi = prefac_xi * np.real(b)

    return r, xi

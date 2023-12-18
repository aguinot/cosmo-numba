""" Simpson integral

Simplified version of the Simpson method from scipy

"""

import numba as nb
import numpy as np


@nb.njit(
    nb.float64[:](nb.float64[:]),
    fastmath=True,
)
def numbadiff(x):
    return x[1:] - x[:-1]


@nb.njit(
    nb.float64(
        nb.float64[:],
        nb.int64,
        nb.int64,
        nb.float64[:],
        nb.float64
    ),
    fastmath=True,
)
def _basic_simpson(y, start, stop, x, dx):

    if start is None:
        start = 0
    step = 2

    if x is None:  # Even-spaced Simpson's rule.
        result = np.sum(
            y[start:stop:step] +
            4.0*y[start+1:stop+1:step] +
            y[start+2:stop+2:step]
        )
        result *= dx / 3.0
    else:

        # Account for possibly different spacings.
        #    Simpson's rule changes a bit.
        h = numbadiff(x)
        h0 = h[start:stop:step]
        h1 = h[start+1:stop+1:step]
        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = np.true_divide(h0, h1)
        tmp = hsum/6.0 * (
            y[start:stop:step] *
            (2.0 - np.true_divide(1.0, h0divh1,)) +
            y[start+1:stop+1:step] *
            (hsum * np.true_divide(hsum, hprod,)) +
            y[start+2:stop+2:step] * (2.0 - h0divh1)
        )
        result = np.sum(tmp)
    return result


@nb.njit(
    [
        nb.float64(nb.float64[:], nb.float64[:], nb.float64),
        nb.float64(nb.float64[:], nb.float64[:], nb.types.Omitted(1.))
    ],
    fastmath=True,
)
def simpson(y, x=None, dx=1.):

    y = np.asarray(y)
    N = y.shape[0]

    if N % 2 == 0:
        val = 0.0
        result = 0.0

        # use Simpson's rule on first intervals
        result = _basic_simpson(y, 0, N-3, x, dx)

        h = np.asfarray([dx, dx])
        if x is not None:
            # grab the last two spacings from the appropriate axis
            diffs = numbadiff(x)
            h = np.array([diffs[-2], diffs[-1]])

        # This is the correction for the last interval according to
        # Cartwright.
        # However, I used the equations given at
        # https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_rule_for_irregularly_spaced_data
        # A footnote on Wikipedia says:
        # Cartwright 2017, Equation 8. The equation in Cartwright is
        # calculating the first interval whereas the equations in the
        # Wikipedia article are adjusting for the last integral. If the
        # proper algebraic substitutions are made, the equation results in
        # the values shown.
        num = 2 * h[1] ** 2 + 3 * h[0] * h[1]
        den = 6 * (h[1] + h[0])
        alpha = np.true_divide(
            num,
            den,
        )

        num = h[1] ** 2 + 3.0 * h[0] * h[1]
        den = 6 * h[0]
        beta = np.true_divide(
            num,
            den,
        )

        num = 1 * h[1] ** 3
        den = 6 * h[0] * (h[0] + h[1])
        eta = np.true_divide(
            num,
            den,
        )

        result += alpha*y[-1] + beta*y[-2] - eta*y[-3]

        result = result + val
    else:
        result = _basic_simpson(y, 0, N-2, x, dx)
    return result

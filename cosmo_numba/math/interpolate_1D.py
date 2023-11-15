""" Interpolation in 1D

This is an implementation of the Akima spline interpolation
Taken from https://github.com/cgohlke/akima adapted to a class and port to
numba.

Author: Axel Guinot

"""
import numpy as np
import numba as nb


spec_akima = [
    ('x', nb.float64[:]),
    ('y', nb.float64[:]),
    ('n', nb.int64),
    ('b', nb.float64[:]),
    ('c', nb.float64[:]),
    ('d', nb.float64[:]),
]


@nb.experimental.jitclass(spec_akima)
class AkimaInterp1D(object):

    def __init__(self, x, y):
        """Return interpolated data using Akima's method.

        This Python implementation is inspired by the Matlab(r) code by
        N. Shamsundar. It lacks certain capabilities of the C implementation
        such as the output array argument and interpolation along an axis of a
        multidimensional data array.

        NOTE:
        Stolen from https://github.com/cgohlke/akima adapted to a class and
        port to numba

        Parameters
        ----------
        x : array like
            1D array of monotonically increasing real values.
        y : array like
            N-D array of real values. y's length along the interpolation
            axis must be equal to the length of x.
        x_new : array like
            New independent variables.
        axis : int
            Specifies axis of y along which to interpolate. Interpolation
            defaults to last axis of y.
        out : array
            Optional array to receive results. Dimension at axis must equal
            length of x.

        Examples
        --------
        >>> interpolate([0, 1, 2], [0, 0, 1], [0.5, 1.5])
        array([-0.125,  0.375])
        >>> x = np.sort(np.random.random(10) * 10)
        >>> y = np.random.normal(0.0, 0.1, size=len(x))
        >>> z = interpolate(x, y, x)
        >>> np.allclose(y, z)
        True
        >>> x = x[:10]
        >>> y = np.reshape(y, (10, -1))
        >>> z = np.reshape(y, (10, -1))
        >>> interpolate(x, y, x, axis=0, out=z)
        >>> np.allclose(y, z)
        True

        """
        self.x = x
        self.y = y

        self.n = len(x)
        if self.n < 3:
            raise ValueError('array too small')
        if self.n != len(y):
            raise ValueError('size of x-array must match data shape')

        dx = np.diff(x)
        if np.any(dx <= 0.0):
            raise ValueError('x-axis not valid')

        m = np.diff(y) / dx
        mm = 2.0 * m[0] - m[1]
        mmm = 2.0 * mm - m[0]
        mp = 2.0 * m[self.n - 2] - m[self.n - 3]
        mpp = 2.0 * mp - m[self.n - 2]

        mm = np.array([mm])
        mmm = np.array([mmm])
        mp = np.array([mp])
        mpp = np.array([mpp])

        m1 = np.concatenate((mmm, mm, m, mp, mpp))

        dm = np.abs(np.diff(m1))
        f1 = dm[2:self.n + 2]
        f2 = dm[0:self.n]
        f12 = f1 + f2

        ids = np.nonzero(f12 > 1e-9 * np.max(f12))[0]
        self.b = m1[1:self.n + 1]

        self.b[ids] = (
            f1[ids] * m1[ids + 1] + f2[ids] * m1[ids + 2]
        ) / f12[ids]
        self.c = (3.0 * m - 2.0 * self.b[0:self.n - 1] - self.b[1:self.n]) / dx
        self.d = (self.b[0:self.n - 1] + self.b[1:self.n] - 2.0 * m) / dx**2

    def eval(self, xout, left=0., right=0.):

        lm = xout < self.x[0]
        rm = xout > self.x[-1]
        totm = np.invert(lm | rm)

        yout = np.zeros_like(xout)

        # Do something better for interpolation
        yout[lm] = left
        yout[rm] = right

        bins = np.digitize(xout[totm], self.x)
        bins = np.minimum(bins, self.n - 1) - 1
        bb = bins[0:len(xout[totm])]
        wj = xout[totm] - self.x[bb]

        yout[totm] = (
            (wj * self.d[bb] + self.c[bb]) * wj + self.b[bb]
        ) * wj + self.y[bb]

        return yout

import os

if os.environ.get("COVERAGE_MODE", "0") == "1":
    os.environ["TESTING_FFTLOG"] = "1"

import numpy as np
from numpy.testing import assert_allclose

import pytest

from scipy.special import gamma

from cosmo_numba.math.integrate.fftlog import fht


def fk(k, alpha):
    """
    Power-law function k^{-alpha}.

    Parameters
    ----------
    k : array_like
        Wavenumber array.
    alpha : float
        Power-law index.

    Returns
    -------
    array_like
        k^{-alpha}
    """
    return k ** (-alpha)


def fr(r, alpha, mu, dim):
    """
    Analytic result of the d-D Hankel transform of k^{-alpha}.

    Parameters
    ----------
    r : array_like
        Distance array.
    alpha : float
        Power-law index.
    mu : int
        Order of the Bessel function in the Hankel transform.
    dim : int
        Dimension of the transform.

    Returns
    -------
    array_like
        Expected result of the Hankel transform
    """
    g1 = gamma(0.5 * (dim - alpha + mu))
    g2 = gamma(0.5 * (alpha + mu))
    den = np.pi ** (dim / 2.0) * 2 ** (alpha)
    return g1 / (g2 * den * r ** (dim - alpha))


@pytest.mark.parametrize("mu", [0, 2])
@pytest.mark.parametrize("alpha", [1.0, 1.2, 1.5, 1.8])
def test_fftlog_plaw(mu, alpha):
    r"""
    Test FFTLog-based Hankel transform of a power law against the analytic
    result.
    The d-D Hankel transform of k^{-alpha} is:
    .. math::
        f(r) = \Gamma[(d - \alpha + \mu) / 2] / \Gamma[(\alpha + \mu) / 2] / (\pi^{d/2} * 2^\alpha * r^{d-\alpha})
    """  # noqa: E501

    dim = 2
    nk_ = 1024
    k_arr = np.logspace(-4, 4, nk_)
    fk_arr = fk(k_arr, alpha)

    r_arr, fr_arr = fht(
        nk_, k_arr, fk_arr, dim, mu, -alpha, kcrc=1.0, noring=1.0
    )

    fr_arr_pred = fr(r_arr, alpha, mu, dim)

    assert_allclose(fr_arr, fr_arr_pred, rtol=1e-10, atol=1e-10)

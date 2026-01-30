import os

if os.environ.get("COVERAGE_MODE", "0") == "1":
    os.environ["TESTING_UTILS"] = "1"

import numpy as np
from numpy.testing import assert_allclose

import pytest

from scipy.special._testutils import FuncData
from scipy.special import gamma

from cosmo_numba.math.utils import lngamma, numbadiff, extend_log_grid, pad_1D


def test_lngamma():
    """
    Test function from Scipy `test_identity1`. The precision has been adjusted
    to account for the implementation used here.
    """
    # test the identity exp(loggamma(z)) = gamma(z)
    x = np.array([-99.5, -9.5, -0.5, 0.5, 9.5, 99.5])
    y = x.copy()
    x, y = np.meshgrid(x, y)
    z = (x + 1j * y).flatten()
    dataset = np.vstack((z, gamma(z))).T

    def f(z):
        return np.exp(lngamma(z))

    FuncData(
        f,
        dataset,
        0,
        1,
        rtol=1e-12,
        atol=1e-12,
        vectorized=False,
    ).check()


@pytest.mark.parametrize(
    "arr",
    [
        np.array([1, 2, 4, 7, 0]),
        np.linspace(0, 10, 100),
        np.random.rand(50),
        np.array([5]),
        np.array([]),
    ],
)
def test_numbadiff(arr):
    """ "
    Test numbadiff against np.diff for various cases.

    Parameters
    ----------
    arr : array_like
        Input array to test.
    """
    result = numbadiff(arr)
    expected = np.diff(arr)
    assert_allclose(result, expected)


@pytest.mark.parametrize(
    "bin_size, pad_low_decades, pad_high_decades",
    [
        (0.3, 0.0, 0.0),
        (0.3, 0.5, 0.5),
        (0.3, 2.0, 3.7),
        (0.0001, 2.0, 2.0),
    ],
)
def test_extend_log_grid(bin_size, pad_low_decades, pad_high_decades):
    """
    Test that the extend_log_grid function extends a logarithmic grid
    correctly.

    Parameters
    ----------
    bin_size : float
        Size of the bins in log space for the original grid.
    pad_low_decades, pad_high_decades : float
        Padding in decades below and above the original grid.
    """
    tmin = 0.5
    tmax = 400
    nbins = int(np.round((np.log(tmax) - np.log(tmin)) / bin_size))
    theta_ = np.logspace(np.log10(tmin), np.log10(tmax), nbins + 1)
    theta = np.mean([theta_[:-1], theta_[1:]], axis=0)

    theta_ext, i_low, i_high = extend_log_grid(
        theta,
        pad_low_decades=pad_low_decades,
        pad_high_decades=pad_high_decades,
    )

    dln = np.median(np.diff(np.log(theta)))

    n_expected_low = int(np.round(pad_low_decades * np.log(10.0) / dln))
    n_expected_high = int(np.round(pad_high_decades * np.log(10.0) / dln))

    expected_theta0 = theta[0] * np.exp(-n_expected_low * dln)
    expected_theta1 = theta[-1] * np.exp(n_expected_high * dln)

    # Check that the lower boud is what we expect
    assert_allclose(
        theta_ext[0],
        expected_theta0,
        rtol=1e-13,
        atol=0.0,
    )
    # Check that the upper bound is what we expect
    assert_allclose(
        theta_ext[-1],
        expected_theta1,
        rtol=1e-13,
        atol=0.0,
    )
    # Check that the original grid is preserved
    assert_allclose(
        theta_ext[i_low:i_high],
        theta,
    )
    # Check that the spacing is the same in the extended grid
    assert_allclose(
        np.diff(np.log(theta_ext)),
        dln,
        rtol=1e-10,
        atol=0.0,
    )


def test_pad_1D_basic():
    """
    Test that the pad_1D function correctly pads a 1D array into an
    extended grid with zeros.
    """
    f_theta = np.array([10.0, 20.0, 30.0, 40.0])
    N_theta_ext = 10
    i_low = 3
    i_high = i_low + f_theta.size

    f_theta_ext = pad_1D(N_theta_ext, i_low, i_high, f_theta)

    # shape
    assert f_theta_ext.shape == (N_theta_ext,)

    # central region copied correctly
    assert_allclose(f_theta_ext[i_low:i_high], f_theta)

    # zero padding
    assert_allclose(f_theta_ext[:i_low], 0.0)
    assert_allclose(f_theta_ext[i_high:], 0.0)

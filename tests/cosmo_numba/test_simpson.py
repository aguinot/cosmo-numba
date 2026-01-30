import os

if os.environ.get("COVERAGE_MODE", "0") == "1":
    os.environ["TESTING_SIMPSON"] = "1"

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from cosmo_numba.math.integrate.simpson import simpson


def test_simpson():
    """
    Simplified version of the Scipy testing suite.
    """

    y = np.arange(17)
    assert_equal(simpson(y), 128)
    assert_equal(simpson(y, dx=0.5), 64)
    assert_equal(simpson(y, x=np.linspace(0, 4, 17)), 32)

    # integral should be exactly 21
    x = np.linspace(1, 4, 4)

    def f(x):
        return x**2

    assert_allclose(simpson(f(x), x=x), 21.0)

    # integral should be exactly 114
    x = np.linspace(1, 7, 4)
    assert_allclose(simpson(f(x), dx=2.0), 114)

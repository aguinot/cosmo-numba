import cosmo_numba


def test_version():
    """Check to see that we can get the package version"""
    assert cosmo_numba.__version__ is not None

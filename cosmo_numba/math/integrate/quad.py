"""Quad integration

Implementation of the quad integration based on NumbaQuadpack
https://github.com/Nicholaswogan/NumbaQuadpack. The dqags has been adapted to
use interpolation which allows the the integration to work on "any" functions.
It can also be called from within jitted functions.

Author: Axel Guinot

"""

import numba as nb
import numpy as np
from NumbaQuadpack import dqags

from itertools import product

from cosmo_numba.math.interpolate.interpolate_1D import (
    nb_interp1d_func,
    spec_interp,
)


def make_signature(output_sig, *args):
    output_comb = list(product(*args, repeat=1))
    sig_final = []
    for output in output_comb:
        sig_tmp = ", ".join(output)
        sig_tmp = f"({sig_tmp})"
        sig_tmp = output_sig + sig_tmp
        sig_final.append(sig_tmp)
    return sig_final


INTERP_ADDRESS = nb.experimental.function_type._get_wrapper_address(
    nb_interp1d_func, spec_interp
)


# Here we use a constructor for the signature due to the high number of
# optional parameters. It would make the code very hard to read otherwise.
# To see the full signature of the function, one can call
# `interp_quad.signature` from python.
spec_interp_quad = make_signature(
    "Tuple((float64, float64, boolean))",
    ["float64"],
    ["float64"],
    ["float64"],
    ["float64[:]"],
    ["float64"],
    ["float64"],
    ["int64", "Omitted(3)"],
    ["boolean", "Omitted(False)"],
    ["boolean", "Omitted(True)"],
    ["int64", "Omitted(0)"],
    ["boolean", "Omitted(False)"],
    ["float64", "Omitted(1.49e-8)"],
    ["float64", "Omitted(1.49e-8)"],
)


@nb.njit(
    # spec_interp_quad,
    fastmath=True,
)
def interp_quad(
    x_start,
    x_end,
    x_step,
    fx,
    a,
    b,
    k=3,
    periodic=False,
    padding=True,
    extrap_dist=0,
    log_interp=False,
    epsabs=1.49e-8,
    epsrel=1.49e-8,
):
    """interp_quad

    Perform an interpolation of degree `k` of the provided array `fx` padded on
    the regular grid `[x_start:x_step:x_end]`. If `fx` is padded on a
    logarithmically spaced grid, one can set `log_interp=True`.

    NOTE: only works on definite bounds and regular grid either in real or log
    space.
    NOTE: For some reason when providing a signature, omitting some parameters
    return an error (`k`, `padding`). Also, not providing the `padding`
    parameter makes the computation longer.

    Parameters
    ----------
    x_start : float
        Grid start.
    x_end : float
        Grid end.
    x_step : _type_
        Grid step.
    fx : numpy.ndarray
        Padded function to integrate.
    a : float
        Lower bound of the integral.
    b : float
        Upper bound of the integral.
    k : int, optional
        Degree of interpolation, by default 3
    periodic : bool, optional
        See interp1d, by default False
    padding : bool, optional
        See interp1d, by default True
    extrap_dist : int, optional
        See interp1d, by default 0
    log_interp : bool, optional
        See interp1d, by default False
    epsabs : float, optional
        Absolute error tolerance, by default 1.49e-8
    epsrel : float, optional
        Relative error tolerance, by default 1.49e-8

    Returns
    -------
    tuple
        Tuple with result, abserr, success.
    """

    len_fx = len(fx)

    data = np.empty(len_fx + 8, dtype=np.float64)
    data[0] = np.float64(len_fx)
    for i in range(len_fx):
        data[i + 1] = fx[i]
    data[len_fx + 1] = x_start
    data[len_fx + 2] = x_end
    data[len_fx + 3] = x_step
    data[len_fx + 4] = np.float64(k)
    data[len_fx + 5] = np.float64(periodic)
    data[len_fx + 6] = np.float64(padding)
    data[len_fx + 7] = np.float64(extrap_dist)
    data[len_fx + 8] = np.float64(log_interp)

    res = dqags(
        INTERP_ADDRESS,
        a,
        b,
        data=data,
        epsabs=epsabs,
        epsrel=epsrel,
    )

    return res

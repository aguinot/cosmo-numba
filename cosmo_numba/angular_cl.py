"""Angular C_ell

Computation of the angular C_ell.
Largely inspired from the CCL package

Author: Axel Guinot

"""

import numpy as np
import numba as nb

from .math.interpolate.interpolate_1D import AkimaInterp1D
from .math.interpolate.interpolate_2D import nb_interp2d
from .math.integrate.simpson import simpson


@nb.njit(
    nb.float64(nb.float64),
    fastmath=True,
)
def get_f_ell(ell):
    """
    From CCL
    ccl_tracers.c
    """
    if ell <= 1:
        return 0.0
    elif ell <= 10:
        return np.sqrt((ell - 1) * (ell) * (ell + 1) * (ell + 2))
    else:
        lp1 = ell + 0.5
        lp1_2 = lp1**2
        if ell <= 1000:
            # This is accurate to 5E-5 for l>10
            return lp1_2 * (1 - 1.25 / lp1_2)
        else:
            # This is accurate to 1E-6 for l>1000
            return lp1_2


@nb.njit(
    nb.float64[:](
        nb.float64,  # ell
        nb.float64[:],  # k
        nb.float64[:],  # a
        nb.float64[:],  # chi
        AkimaInterp1D.class_type.instance_type,  # tracer 2
    ),
    fastmath=True,
)
def get_transfer_limber(ell, k, a, chi, tracer_interp):
    w = tracer_interp.eval(chi)
    fl = get_f_ell(ell)

    return w * fl / (ell + 0.5) ** 2


@nb.njit(
    nb.float64[:](
        nb.float64,  # ell
        nb.float64[:],  # lk
        AkimaInterp1D.class_type.instance_type,  # chi2a_interp
        nb_interp2d.class_type.instance_type,  # pk_interp
        AkimaInterp1D.class_type.instance_type,  # tracer1_interp
        AkimaInterp1D.class_type.instance_type,  # tracer2_interp
    ),
    fastmath=True,
)
def cl_integrand(
    ell, lk, chi2a_interp, pk_interp, tracer1_interp, tracer2_interp
):
    k = np.exp(lk)
    chi = (ell + 0.5) / k
    a = chi2a_interp.eval(chi)

    pk = pk_interp.eval(a, lk)

    t1 = get_transfer_limber(ell, k, a, chi, tracer1_interp)
    t2 = get_transfer_limber(ell, k, a, chi, tracer2_interp)

    return k * pk * t1 * t2


@nb.njit(
    nb.float64(
        nb.float64,  # ell
        nb.float64,  # lkmin
        nb.float64,  # lkmax,
        AkimaInterp1D.class_type.instance_type,  # chi2a_interp
        nb_interp2d.class_type.instance_type,  # pk_interp
        AkimaInterp1D.class_type.instance_type,  # tracer1_interp
        AkimaInterp1D.class_type.instance_type,  # tracer2_interp
    ),
    fastmath=True,
)
def get_angular_1ell(
    ell, lkmin, lkmax, chi2a_interp, pk_interp, tracer1_interp, tracer2_interp
):
    dlogk_integ = 0.025

    nk = int(max((lkmax - lkmin) / dlogk_integ + 0.5, 1) + 1)
    lk_arr = np.linspace(lkmin, lkmax, nk)

    integ = cl_integrand(
        ell, lk_arr, chi2a_interp, pk_interp, tracer1_interp, tracer2_interp
    )
    return simpson(integ, lk_arr)


@nb.njit(
    nb.types.Tuple((nb.float64, nb.float64))(
        nb.float64[:], nb.float64[:], nb.float64
    ),
    fastmath=True,
)
def get_k_bound(chi1, chi2, ell):
    chi_min = max(np.min(chi1), np.min(chi2))
    chi_max = min(np.max(chi1), np.max(chi2))

    if chi_min <= 0.0:
        chi_min = 0.5 * (ell + 0.5) / 1000.0

    lkmax = np.log(min(1000.0, 2 * (ell + 0.5) / chi_min))
    lkmin = np.log(max(5e-5, (ell + 0.5) / chi_max))
    return (lkmin, lkmax)


@nb.njit(
    nb.void(
        nb.float64[:],
        AkimaInterp1D.class_type.instance_type,
        nb.float64[:, :],
        nb.float64[:, :],
        nb_interp2d.class_type.instance_type,
        nb.float64[:],
    ),
    fastmath=True,
)
def cl_integ(ell_arr, chi2a_interp, tracer1, tracer2, pk_interp, cl):
    tracer1_interp = AkimaInterp1D(
        tracer1[0].astype("float64"), tracer1[1].astype("float64")
    )
    tracer2_interp = AkimaInterp1D(
        tracer2[0].astype("float64"), tracer2[1].astype("float64")
    )

    for i in range(len(ell_arr)):
        lkmin, lkmax = get_k_bound(tracer1[0], tracer2[0], ell_arr[i])
        cl[i] = get_angular_1ell(
            ell_arr[i],
            lkmin,
            lkmax,
            chi2a_interp,
            pk_interp,
            tracer1_interp,
            tracer2_interp,
        ) / (ell_arr[i] + 0.5)

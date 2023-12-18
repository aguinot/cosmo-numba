""" Probes Kernel

Author: Axel Guinot

Kernel currently implemented:
- Lensing

"""
import numpy as np
import numba as nb

from cosmo_numba.math.integrate.simpson import simpson
from cosmo_numba.math.interpolate.interpolate_1D import AkimaInterp1D


@nb.njit(
    nb.float64[:](
        nb.float64[:],
        nb.float64[:],
        nb.float64
    ),
    fastmath=True,
)
def integrand(dndz, chi_prime, chi):
    return dndz*np.clip((chi_prime-chi), 0, None)/np.clip(chi_prime, 1, None)


@nb.njit(
    nb.float64(
        nb.float64[:],
        nb.float64,
        nb.float64[:],
        nb.float64,
        nb.float64,
        nb.float64[:]
    ),
    fastmath=True
)
def lensing_int_kernel(dndz, z, z_prime, z_max, chi, chi_prime):

    integ = integrand(dndz, chi_prime, chi)
    res = simpson(integ, z_prime)

    return res * chi * (1 + z)


@nb.njit(
    [
        nb.void(
            AkimaInterp1D.class_type.instance_type,
            AkimaInterp1D.class_type.instance_type,
            nb.float64[:],
            nb.float64,
            nb.float64[:],
            nb.float64,
            nb.float64[:],
            nb.types.Omitted(256)
        ),
        nb.void(
            AkimaInterp1D.class_type.instance_type,
            AkimaInterp1D.class_type.instance_type,
            nb.float64[:],
            nb.float64,
            nb.float64[:],
            nb.float64,
            nb.float64[:],
            nb.int64,
        ),
    ]
)
def f_int_kernel(
    z2chi,
    dndz_interp,
    z_arr, z_max,
    chi_arr, chi_max,
    int_kernel,
    n_samp=256,
):
    for i, (z_, chi_) in enumerate(zip(z_arr, chi_arr)):
        z_tmp = np.linspace(z_, z_max, n_samp)

        chi_tmp = z2chi.eval(z_tmp)
        dndz_ = dndz_interp.eval(z_tmp)

        int_kernel[i] = lensing_int_kernel(
            dndz_,
            z_,
            z_tmp,
            z_max,
            chi_,
            chi_tmp
        )

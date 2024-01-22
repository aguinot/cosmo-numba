""" COSEBIs

Compute COSEBIs based on Schneider et al. 2010
(https://arxiv.org/abs/1002.2136).
This code include the computation of the roots and normalizations requiered to
compute T_plus_log. At the moment the code can only compute the "log" quantity.
The code has been tested against Mathematica for the roots/norm and against
nicaea for the COSEBIs E-/B-modes.

Author: Axel Guinot

"""
import numpy as np
import sympy as sym
import mpmath as mp
import numba as nb

from cosmo_numba.B_modes.cosebis_nb import (
    tp_n_log,
    tm_n_log,
    wn_log,
    get_xipm_cosebis,
    get_Cell_cosebis,
    get_cosebis_cov_from_xipm_cov,
)


class COSEBIS():

    def __init__(self, theta_min, theta_max, N_max, precision=80):

        mp.mp.dps = precision

        self.tmin = theta_min
        self.tmax = theta_max
        self.Nmax = N_max

    def _compute_roots(self, use_nsolve=False):
        """compute roots

        Compute the roots, normalization and Tp_log used in the COSEBIs
        computation.
        This code is derived from the mathematica version proposed in Fig. 3 of
        Schneider et al. 2010 (https://arxiv.org/abs/1002.2136).
        The equations referenced in the code come from the paper above.

        Parameters
        ----------
        use_nsolve : bool, optional
            use `nsolve` (slower) instead of `nroots`, by default False.
        """

        z = sym.Symbol("z")
        i = sym.Symbol("i")

        # Eq. 29 for theta = theta_max
        zm = sym.N(
            sym.log(
                sym.nsimplify(
                    self.tmax/self.tmin,
                    rational=True,
                    # tolerance=1e-10,
                    tolerance=1e-20,
                )
            ),
            130,
        )

        # Eq 32, k=[1,2]
        J = sym.Array(
            [
                [
                    sym.re(
                        sym.lowergamma(j+1, -(k+1)*zm)/((-(k+1))**(j+1)),
                    )
                    for j in range(0, 2*self.Nmax+1+1)
                ]
                for k in range(0, 2)
            ]
        ).as_mutable()

        # Eq 32, k=4
        J4 = sym.Array(
            [
                sym.re(
                    sym.lowergamma(j+1, -4*zm)/((-4)**(j+1)),
                )
                for j in range(0, 2*self.Nmax+1+1)
            ]
        ).as_mutable()

        # Those arrays are not accessed during the first iteration, we
        # initialized them just to have a nice code.
        c = sym.Array([0]).as_mutable()
        c_old = sym.Array([0]).as_mutable()

        all_roots = []
        all_norm = []
        tp_log = []
        for n in range(1, self.Nmax+1):
            a = sym.Array([[0]*(n+1)]*(n+1)).as_mutable()
            for j in range(0, n+1):
                a[n-1, j] = J[2-1, j]/J[2-1, n+1]
                a[n+1-1, j] = J4[j]/J4[n+1]
            b = sym.Array([0]*(n+1)).as_mutable()
            b[n-1,] = -1
            b[n+1-1,] = -1

            for m in range(1, n):
                for j in range(0, n+1):
                    tmp_sum = 0
                    for ii in range(0, m+1+1):
                        tmp_sum += J[0, ii+j]*c[m-1, ii]
                    a[m-1, j] = tmp_sum

            bb = sym.Array([0]*(n)).as_mutable()
            for m in range(1, n):
                tmp_sum = 0
                for ii in range(0, m+1+1):
                    tmp_sum += J[0, ii+n+1]*c[m-1, ii]
                bb[m-1,] = -tmp_sum

            for m in range(1, n):
                for j in range(0, n+1):
                    a[m-1, j] = a[m-1, j]/bb[m-1]
            for m in range(1, n):
                b[m-1,] = sym.Float(1)

            A = sym.Matrix(
                [[a[i-1, j] for j in range(0, n+1)] for i in range(1, n+1+1)]
            )
            B = sym.Matrix([b[i-1] for i in range(1, n+1+1)])

            CC = sym.linsolve((A, B))

            c = sym.Array([[0]*(n+1+1)]*(n)).as_mutable()
            for ii in range(n-1):
                for j in range(n+1):
                    c[ii, j] = c_old[ii, j]

            for j in range(0, n+1):
                c[n-1, j] = CC.args[0][j]
            c[n-1, n+1] = 1
            c_old = c

            # Eq. 28
            tt = sym.summation(c[n-1, i]*z**(i), (i, 0, n+1))
            if use_nsolve:
                # Precision drop here for some reason..
                # We then use that as an init for nsolve ()
                roots_init = sym.solve(tt, z, rational=False)
                roots = [sym.nsolve(tt, r_tmp) for r_tmp in roots_init]
            else:
                roots = sym.nroots(tt, n=mp.mp.dps)

            r = sym.Array([sym.N(sym.re(roots[j]), 50) for j in range(0, n+1)])
            # Eq. 36 (without normalization)
            t = sym.product(z-r[i,], (i, 0, n))

            # Eq. 31
            mp_integrand = sym.lambdify(z, sym.exp(z)*t**2, modules="mpmath")
            mp_res = mp.quad(mp_integrand, [0, zm])
            normgral = sym.sympify(mp_res)

            norm = sym.sqrt((sym.exp(zm)-1)/normgral)

            all_roots.append(r)
            all_norm.append(sym.N(norm, 50))
            tp_log.append(t*sym.N(norm, 50))

        self.roots = nb.typed.List([
            np.array([
                np.float64(roots_tmp)
                for roots_tmp in roots_n_tmp
            ])
            for roots_n_tmp in all_roots
        ])
        self.norms = np.array([
            np.float64(norm) for norm in all_norm
        ])

    def get_Tp_log(self, theta):

        if not hasattr(self, "roots"):
            self._compute_roots()

        z = np.log(theta/self.tmin)
        Tp_log = tp_n_log(z, self.roots, self.norms)

        return Tp_log

    def get_Tm_log(self, theta):

        if not hasattr(self, "roots"):
            self._compute_roots()

        z = np.log(theta/self.tmin)
        Tm_log = tm_n_log(z, self.roots, self.norms)

        return Tm_log

    def get_Wn_log(self, theta=None, q=-1.1):

        if theta is None:
            theta = np.logspace(
                np.log10(1.),
                np.log10(400.),
                5_000
            )

        Tp_log = self.get_Tp_log(theta)

        theta_rad = np.deg2rad(theta/60)

        Wn_log = wn_log(theta_rad, Tp_log, q)

        return Wn_log

    def cosebis_from_xipm(self, theta, dtheta, xi_plus, xi_minus):

        if (np.min(theta) < self.tmin) | (np.max(theta) > self.tmax):
            print(
                "WARNING: The range of theta for xipm is outside the range "
                "used to compute the roots and normalizations "
                f"[{self.tmin, self.tmax}]."
            )

        Tp_log = self.get_Tp_log(theta)
        Tm_log = self.get_Tm_log(theta)

        theta_rad = np.deg2rad(theta/60)
        dtheta_rad = np.deg2rad(dtheta/60)
        C_E, C_B = get_xipm_cosebis(
            theta_rad,
            dtheta_rad,
            xi_plus,
            xi_minus,
            Tp_log,
            Tm_log,
            self.Nmax
        )

        return C_E, C_B

    def cosebis_from_Cell(self, ell, Cell_E, Cell_B, theta=None):

        Wn_log = self.get_Wn_log(theta)

        C_E, C_B = get_Cell_cosebis(
            ell,
            Cell_E,
            Cell_B,
            Wn_log,
            self.Nmax
        )

        return C_E, C_B

    def cosebis_covariance_from_xipm_covariance(
        self, theta, dtheta, cov_xi_pm
    ):

        if (np.min(theta) < self.tmin) | (np.max(theta) > self.tmax):
            print(
                "WARNING: The range of theta for xipm is outside the range "
                "used to compute the roots and normalizations "
                f"[{self.tmin, self.tmax}]."
            )

        Tp_log = self.get_Tp_log(theta)
        Tm_log = self.get_Tm_log(theta)

        theta_rad = np.deg2rad(theta/60)
        dtheta_rad = np.deg2rad(dtheta/60)
        Cov_EB = get_cosebis_cov_from_xipm_cov(
            theta_rad,
            dtheta_rad,
            cov_xi_pm,
            Tp_log,
            Tm_log,
            self.Nmax
        )

        return Cov_EB

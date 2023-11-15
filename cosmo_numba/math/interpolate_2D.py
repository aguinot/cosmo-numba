import numpy as np
import numba as nb
# from numba import int64, float64, boolean, typeof


@nb.njit(
    nb.void(
        nb.float64[:, :],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.int64[:],
        nb.boolean[:],
        nb.int64[:],
        nb.float64[:],
        nb.float64[:]
    ),
    fastmath=True,
)
def _interp2d_k1(f, xout, yout, fout, a, h, n, p, o, lb, ub):
    m = fout.shape[0]
    for mi in nb.prange(m):
        xr = min(max(xout[mi], lb[0]), ub[0])
        yr = min(max(yout[mi], lb[1]), ub[1])
        xx = xr - a[0]
        yy = yr - a[1]
        ix = int(xx//h[0])
        iy = int(yy//h[1])
        ratx = xx/h[0] - (ix+0.5)
        raty = yy/h[1] - (iy+0.5)
        asx = np.empty(2)
        asy = np.empty(2)
        asx[0] = 0.5 - ratx
        asx[1] = 0.5 + ratx
        asy[0] = 0.5 - raty
        asy[1] = 0.5 + raty
        ix += o[0]
        iy += o[1]
        fout[mi] = 0.0
        for i in range(2):
            ixi = (ix + i) % n[0] if p[0] else ix + i
            for j in range(2):
                iyj = (iy + j) % n[1] if p[1] else iy + j
                fout[mi] += f[ixi, iyj]*asx[i]*asy[j]


@nb.njit(
    nb.void(
        nb.float64[:, :],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.int64[:],
        nb.boolean[:],
        nb.int64[:],
        nb.float64[:],
        nb.float64[:]
    ),
    fastmath=True,
)
def _interp2d_k3(f, xout, yout, fout, a, h, n, p, o, lb, ub):
    m = fout.shape[0]
    for mi in nb.prange(m):
        xr = min(max(xout[mi], lb[0]), ub[0])
        yr = min(max(yout[mi], lb[1]), ub[1])
        xx = xr - a[0]
        yy = yr - a[1]
        ix = int(xx//h[0])
        iy = int(yy//h[1])
        ratx = xx/h[0] - (ix+0.5)
        raty = yy/h[1] - (iy+0.5)
        asx = np.empty(4)
        asy = np.empty(4)
        asx[0] = -1/16 + ratx*( 1/24 + ratx*( 1/4 - ratx/6))  # noqa
        asx[1] =  9/16 + ratx*( -9/8 + ratx*(-1/4 + ratx/2))  # noqa
        asx[2] =  9/16 + ratx*(  9/8 + ratx*(-1/4 - ratx/2))  # noqa
        asx[3] = -1/16 + ratx*(-1/24 + ratx*( 1/4 + ratx/6))  # noqa
        asy[0] = -1/16 + raty*( 1/24 + raty*( 1/4 - raty/6))  # noqa
        asy[1] =  9/16 + raty*( -9/8 + raty*(-1/4 + raty/2))  # noqa
        asy[2] =  9/16 + raty*(  9/8 + raty*(-1/4 - raty/2))  # noqa
        asy[3] = -1/16 + raty*(-1/24 + raty*( 1/4 + raty/6))  # noqa
        ix += o[0]-1
        iy += o[1]-1
        fout[mi] = 0.0
        for i in range(4):
            ixi = (ix + i) % n[0] if p[0] else ix + i
            for j in range(4):
                iyj = (iy + j) % n[1] if p[1] else iy + j
                fout[mi] += f[ixi, iyj]*asx[i]*asy[j]


@nb.njit(
    nb.void(
        nb.float64[:, :],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.int64[:],
        nb.boolean[:],
        nb.int64[:],
        nb.float64[:],
        nb.float64[:]
    ),
    fastmath=True,
)
def _interp2d_k5(f, xout, yout, fout, a, h, n, p, o, lb, ub):
    m = fout.shape[0]
    for mi in nb.prange(m):
        xr = min(max(xout[mi], lb[0]), ub[0])
        yr = min(max(yout[mi], lb[1]), ub[1])
        xx = xr - a[0]
        yy = yr - a[1]
        ix = int(xx//h[0])
        iy = int(yy//h[1])
        ratx = xx/h[0] - (ix+0.5)
        raty = yy/h[1] - (iy+0.5)
        asx = np.empty(6)
        asy = np.empty(6)
        asx[0] =   3/256 + ratx*(   -9/1920 + ratx*( -5/48/2 + ratx*(  1/8/6 + ratx*( 1/2/24 -  1/8/120*ratx))))  # noqa
        asx[1] = -25/256 + ratx*(  125/1920 + ratx*( 39/48/2 + ratx*(-13/8/6 + ratx*(-3/2/24 +  5/8/120*ratx))))  # noqa
        asx[2] = 150/256 + ratx*(-2250/1920 + ratx*(-34/48/2 + ratx*( 34/8/6 + ratx*( 2/2/24 - 10/8/120*ratx))))  # noqa
        asx[3] = 150/256 + ratx*( 2250/1920 + ratx*(-34/48/2 + ratx*(-34/8/6 + ratx*( 2/2/24 + 10/8/120*ratx))))  # noqa
        asx[4] = -25/256 + ratx*( -125/1920 + ratx*( 39/48/2 + ratx*( 13/8/6 + ratx*(-3/2/24 -  5/8/120*ratx))))  # noqa
        asx[5] =   3/256 + ratx*(    9/1920 + ratx*( -5/48/2 + ratx*( -1/8/6 + ratx*( 1/2/24 +  1/8/120*ratx))))  # noqa
        asy[0] =   3/256 + raty*(   -9/1920 + raty*( -5/48/2 + raty*(  1/8/6 + raty*( 1/2/24 -  1/8/120*raty))))  # noqa
        asy[1] = -25/256 + raty*(  125/1920 + raty*( 39/48/2 + raty*(-13/8/6 + raty*(-3/2/24 +  5/8/120*raty))))  # noqa
        asy[2] = 150/256 + raty*(-2250/1920 + raty*(-34/48/2 + raty*( 34/8/6 + raty*( 2/2/24 - 10/8/120*raty))))  # noqa
        asy[3] = 150/256 + raty*( 2250/1920 + raty*(-34/48/2 + raty*(-34/8/6 + raty*( 2/2/24 + 10/8/120*raty))))  # noqa
        asy[4] = -25/256 + raty*( -125/1920 + raty*( 39/48/2 + raty*( 13/8/6 + raty*(-3/2/24 -  5/8/120*raty))))  # noqa
        asy[5] =   3/256 + raty*(    9/1920 + raty*( -5/48/2 + raty*( -1/8/6 + raty*( 1/2/24 +  1/8/120*raty))))  # noqa
        ix += o[0]-2
        iy += o[1]-2
        fout[mi] = 0.0
        for i in range(6):
            ixi = (ix + i) % n[0] if p[0] else ix + i
            for j in range(6):
                iyj = (iy + j) % n[1] if p[1] else iy + j
                fout[mi] += f[ixi, iyj]*asx[i]*asy[j]


@nb.njit(
    nb.void(
        nb.float64[:, :],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.int64[:],
        nb.boolean[:],
        nb.int64[:],
        nb.float64[:],
        nb.float64[:]
    ),
    fastmath=True,
)
def _interp2d_k7(f, xout, yout, fout, a, h, n, p, o, lb, ub):
    m = fout.shape[0]
    for mi in nb.prange(m):
        xr = min(max(xout[mi], lb[0]), ub[0])
        yr = min(max(yout[mi], lb[1]), ub[1])
        xx = xr - a[0]
        yy = yr - a[1]
        ix = int(xx//h[0])
        iy = int(yy//h[1])
        ratx = xx/h[0] - (ix+0.5)
        raty = yy/h[1] - (iy+0.5)
        asx = np.empty(8)
        asy = np.empty(8)
        asx[0] =   -5/2048 + ratx*(     75/107520 + ratx*(  259/11520/2 + ratx*(  -37/1920/6 + ratx*(  -7/48/24 + ratx*(   5/24/120 + ratx*( 1/2/720 -  1/5040*ratx))))))  # noqa
        asx[1] =   49/2048 + ratx*(  -1029/107520 + ratx*(-2495/11520/2 + ratx*(  499/1920/6 + ratx*(  59/48/24 + ratx*( -59/24/120 + ratx*(-5/2/720 +  7/5040*ratx))))))  # noqa
        asx[2] = -245/2048 + ratx*(   8575/107520 + ratx*(11691/11520/2 + ratx*(-3897/1920/6 + ratx*(-135/48/24 + ratx*( 225/24/120 + ratx*( 9/2/720 - 21/5040*ratx))))))  # noqa
        asx[3] = 1225/2048 + ratx*(-128625/107520 + ratx*(-9455/11520/2 + ratx*( 9455/1920/6 + ratx*(  83/48/24 + ratx*(-415/24/120 + ratx*(-5/2/720 + 35/5040*ratx))))))  # noqa
        asx[4] = 1225/2048 + ratx*( 128625/107520 + ratx*(-9455/11520/2 + ratx*(-9455/1920/6 + ratx*(  83/48/24 + ratx*( 415/24/120 + ratx*(-5/2/720 - 35/5040*ratx))))))  # noqa
        asx[5] = -245/2048 + ratx*(  -8575/107520 + ratx*(11691/11520/2 + ratx*( 3897/1920/6 + ratx*(-135/48/24 + ratx*(-225/24/120 + ratx*( 9/2/720 + 21/5040*ratx))))))  # noqa
        asx[6] =   49/2048 + ratx*(   1029/107520 + ratx*(-2495/11520/2 + ratx*( -499/1920/6 + ratx*(  59/48/24 + ratx*(  59/24/120 + ratx*(-5/2/720 -  7/5040*ratx))))))  # noqa
        asx[7] =   -5/2048 + ratx*(    -75/107520 + ratx*(  259/11520/2 + ratx*(   37/1920/6 + ratx*(  -7/48/24 + ratx*(  -5/24/120 + ratx*( 1/2/720 +  1/5040*ratx))))))  # noqa
        asy[0] =   -5/2048 + raty*(     75/107520 + raty*(  259/11520/2 + raty*(  -37/1920/6 + raty*(  -7/48/24 + raty*(   5/24/120 + raty*( 1/2/720 -  1/5040*raty))))))  # noqa
        asy[1] =   49/2048 + raty*(  -1029/107520 + raty*(-2495/11520/2 + raty*(  499/1920/6 + raty*(  59/48/24 + raty*( -59/24/120 + raty*(-5/2/720 +  7/5040*raty))))))  # noqa
        asy[2] = -245/2048 + raty*(   8575/107520 + raty*(11691/11520/2 + raty*(-3897/1920/6 + raty*(-135/48/24 + raty*( 225/24/120 + raty*( 9/2/720 - 21/5040*raty))))))  # noqa
        asy[3] = 1225/2048 + raty*(-128625/107520 + raty*(-9455/11520/2 + raty*( 9455/1920/6 + raty*(  83/48/24 + raty*(-415/24/120 + raty*(-5/2/720 + 35/5040*raty))))))  # noqa
        asy[4] = 1225/2048 + raty*( 128625/107520 + raty*(-9455/11520/2 + raty*(-9455/1920/6 + raty*(  83/48/24 + raty*( 415/24/120 + raty*(-5/2/720 - 35/5040*raty))))))  # noqa
        asy[5] = -245/2048 + raty*(  -8575/107520 + raty*(11691/11520/2 + raty*( 3897/1920/6 + raty*(-135/48/24 + raty*(-225/24/120 + raty*( 9/2/720 + 21/5040*raty))))))  # noqa
        asy[6] =   49/2048 + raty*(   1029/107520 + raty*(-2495/11520/2 + raty*( -499/1920/6 + raty*(  59/48/24 + raty*(  59/24/120 + raty*(-5/2/720 -  7/5040*raty))))))  # noqa
        asy[7] =   -5/2048 + raty*(    -75/107520 + raty*(  259/11520/2 + raty*(   37/1920/6 + raty*(  -7/48/24 + raty*(  -5/24/120 + raty*( 1/2/720 +  1/5040*raty))))))  # noqa
        ix += o[0]-3
        iy += o[1]-3
        fout[mi] = 0.0
        for i in range(8):
            ixi = (ix + i) % n[0] if p[0] else ix + i
            for j in range(8):
                iyj = (iy + j) % n[1] if p[1] else iy + j
                fout[mi] += f[ixi, iyj]*asx[i]*asy[j]


@nb.njit(
    nb.void(
        nb.float64[:, :],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.int64[:],
        nb.boolean[:],
        nb.int64[:],
        nb.float64[:],
        nb.float64[:]
    ),
    fastmath=True,
)
def _interp2d_k9(f, xout, yout, fout, a, h, n, p, o, lb, ub):
    m = fout.shape[0]
    for mi in nb.prange(m):
        xr = min(max(xout[mi], lb[0]), ub[0])
        yr = min(max(yout[mi], lb[1]), ub[1])
        xx = xr - a[0]
        yy = yr - a[1]
        ix = int(xx//h[0])
        iy = int(yy//h[1])
        ratx = xx/h[0] - (ix+0.5)
        raty = yy/h[1] - (iy+0.5)
        asx = np.empty(10)
        asy = np.empty(10)
        asx[0] =    35/65536 + ratx*(    -1225/10321920 + ratx*(  -3229/645120/2 + ratx*(    3229/967680/6 + ratx*(   141/3840/24 + ratx*(   -47/1152/120 + ratx*(  -3/16/720 + ratx*(    7/24/5040 + ratx*(  1/2/40320 -   1/362880*ratx))))))))  # noqa
        asx[1] =  -405/65536 + ratx*(    18225/10321920 + ratx*(  37107/645120/2 + ratx*(  -47709/967680/6 + ratx*( -1547/3840/24 + ratx*(   663/1152/120 + ratx*(  29/16/720 + ratx*(  -87/24/5040 + ratx*( -7/2/40320 +   9/362880*ratx))))))))  # noqa
        asx[2] =  2268/65536 + ratx*(  -142884/10321920 + ratx*(-204300/645120/2 + ratx*(  367740/967680/6 + ratx*(  7540/3840/24 + ratx*( -4524/1152/120 + ratx*(-100/16/720 + ratx*(  420/24/5040 + ratx*( 20/2/40320 -  36/362880*ratx))))))))  # noqa
        asx[3] = -8820/65536 + ratx*(   926100/10321920 + ratx*( 745108/645120/2 + ratx*(-2235324/967680/6 + ratx*(-14748/3840/24 + ratx*( 14748/1152/120 + ratx*( 156/16/720 + ratx*(-1092/24/5040 + ratx*(-28/2/40320 +  84/362880*ratx))))))))  # noqa
        asx[4] = 39690/65536 + ratx*(-12502350/10321920 + ratx*(-574686/645120/2 + ratx*( 5172174/967680/6 + ratx*(  8614/3840/24 + ratx*(-25842/1152/120 + ratx*( -82/16/720 + ratx*( 1722/24/5040 + ratx*( 14/2/40320 - 126/362880*ratx))))))))  # noqa
        asx[5] = 39690/65536 + ratx*( 12502350/10321920 + ratx*(-574686/645120/2 + ratx*(-5172174/967680/6 + ratx*(  8614/3840/24 + ratx*( 25842/1152/120 + ratx*( -82/16/720 + ratx*(-1722/24/5040 + ratx*( 14/2/40320 + 126/362880*ratx))))))))  # noqa
        asx[6] = -8820/65536 + ratx*(  -926100/10321920 + ratx*( 745108/645120/2 + ratx*( 2235324/967680/6 + ratx*(-14748/3840/24 + ratx*(-14748/1152/120 + ratx*( 156/16/720 + ratx*( 1092/24/5040 + ratx*(-28/2/40320 -  84/362880*ratx))))))))  # noqa
        asx[7] =  2268/65536 + ratx*(   142884/10321920 + ratx*(-204300/645120/2 + ratx*( -367740/967680/6 + ratx*(  7540/3840/24 + ratx*(  4524/1152/120 + ratx*(-100/16/720 + ratx*( -420/24/5040 + ratx*( 20/2/40320 +  36/362880*ratx))))))))  # noqa
        asx[8] =  -405/65536 + ratx*(   -18225/10321920 + ratx*(  37107/645120/2 + ratx*(   47709/967680/6 + ratx*( -1547/3840/24 + ratx*(  -663/1152/120 + ratx*(  29/16/720 + ratx*(   87/24/5040 + ratx*( -7/2/40320 -   9/362880*ratx))))))))  # noqa
        asx[9] =    35/65536 + ratx*(     1225/10321920 + ratx*(  -3229/645120/2 + ratx*(   -3229/967680/6 + ratx*(   141/3840/24 + ratx*(    47/1152/120 + ratx*(  -3/16/720 + ratx*(   -7/24/5040 + ratx*(  1/2/40320 +   1/362880*ratx))))))))  # noqa
        asy[0] =    35/65536 + raty*(    -1225/10321920 + raty*(  -3229/645120/2 + raty*(    3229/967680/6 + raty*(   141/3840/24 + raty*(   -47/1152/120 + raty*(  -3/16/720 + raty*(    7/24/5040 + raty*(  1/2/40320 -   1/362880*raty))))))))  # noqa
        asy[1] =  -405/65536 + raty*(    18225/10321920 + raty*(  37107/645120/2 + raty*(  -47709/967680/6 + raty*( -1547/3840/24 + raty*(   663/1152/120 + raty*(  29/16/720 + raty*(  -87/24/5040 + raty*( -7/2/40320 +   9/362880*raty))))))))  # noqa
        asy[2] =  2268/65536 + raty*(  -142884/10321920 + raty*(-204300/645120/2 + raty*(  367740/967680/6 + raty*(  7540/3840/24 + raty*( -4524/1152/120 + raty*(-100/16/720 + raty*(  420/24/5040 + raty*( 20/2/40320 -  36/362880*raty))))))))  # noqa
        asy[3] = -8820/65536 + raty*(   926100/10321920 + raty*( 745108/645120/2 + raty*(-2235324/967680/6 + raty*(-14748/3840/24 + raty*( 14748/1152/120 + raty*( 156/16/720 + raty*(-1092/24/5040 + raty*(-28/2/40320 +  84/362880*raty))))))))  # noqa
        asy[4] = 39690/65536 + raty*(-12502350/10321920 + raty*(-574686/645120/2 + raty*( 5172174/967680/6 + raty*(  8614/3840/24 + raty*(-25842/1152/120 + raty*( -82/16/720 + raty*( 1722/24/5040 + raty*( 14/2/40320 - 126/362880*raty))))))))  # noqa
        asy[5] = 39690/65536 + raty*( 12502350/10321920 + raty*(-574686/645120/2 + raty*(-5172174/967680/6 + raty*(  8614/3840/24 + raty*( 25842/1152/120 + raty*( -82/16/720 + raty*(-1722/24/5040 + raty*( 14/2/40320 + 126/362880*raty))))))))  # noqa
        asy[6] = -8820/65536 + raty*(  -926100/10321920 + raty*( 745108/645120/2 + raty*( 2235324/967680/6 + raty*(-14748/3840/24 + raty*(-14748/1152/120 + raty*( 156/16/720 + raty*( 1092/24/5040 + raty*(-28/2/40320 -  84/362880*raty))))))))  # noqa
        asy[7] =  2268/65536 + raty*(   142884/10321920 + raty*(-204300/645120/2 + raty*( -367740/967680/6 + raty*(  7540/3840/24 + raty*(  4524/1152/120 + raty*(-100/16/720 + raty*( -420/24/5040 + raty*( 20/2/40320 +  36/362880*raty))))))))  # noqa
        asy[8] =  -405/65536 + raty*(   -18225/10321920 + raty*(  37107/645120/2 + raty*(   47709/967680/6 + raty*( -1547/3840/24 + raty*(  -663/1152/120 + raty*(  29/16/720 + raty*(   87/24/5040 + raty*( -7/2/40320 -   9/362880*raty))))))))  # noqa
        asy[9] =    35/65536 + raty*(     1225/10321920 + raty*(  -3229/645120/2 + raty*(   -3229/967680/6 + raty*(   141/3840/24 + raty*(    47/1152/120 + raty*(  -3/16/720 + raty*(   -7/24/5040 + raty*(  1/2/40320 +   1/362880*raty))))))))  # noqa
        ix += o[0]-4
        iy += o[1]-4
        fout[mi] = 0.0
        for i in range(10):
            ixi = (ix + i) % n[0] if p[0] else ix + i
            for j in range(10):
                iyj = (iy + j) % n[1] if p[1] else iy + j
                fout[mi] += f[ixi, iyj]*asx[i]*asy[j]


@nb.njit(
    nb.types.Tuple((nb.float64, nb.float64))(
        nb.float64,
        nb.float64,
        nb.float64,
        nb.boolean,
        nb.boolean,
        nb.int64,
        nb.int64,
    ),
    fastmath=True,
)
def _compute_bounds1(a, b, h, p, c, e, k):
    if p:
        return -1e100, 1e100
    elif not c:
        d = h*(k//2)
        return a+d, b-d
    else:
        d = e*h
        u = b+d
        # the routines can fail when we exactly hit the right endpoint,
        # this protects against that
        u -= u*1e-15
        return a-d, u


@nb.njit(
    nb.types.Tuple((nb.float64[:], nb.float64[:]))(
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.boolean[:],
        nb.boolean[:],
        nb.int64[:],
        nb.int64,
    ),
    fastmath=True,
)
def _compute_bounds(a, b, h, p, c, e, k):
    m = len(a)
    bounds = [
        _compute_bounds1(a[i], b[i], h[i], p[i], c[i], e[i], k)
        for i in range(m)
    ]
    return (
        np.array([bounds[0][0], bounds[1][0]]),
        np.array([bounds[0][1], bounds[1][1]])
    )


@nb.njit(
    nb.void(
        nb.float64[:, :],
        nb.int64,
        nb.int64,
    ),
    fastmath=True,
)
def _extrapolate1d_x(f, k, o):
    for ix in range(o):
        il = o-ix-1
        ih = f.shape[0]-(o-ix)
        if k == 1:
            f[il] = 2*f[il+1] - 1*f[il+2]
            f[ih] = 2*f[ih-1] - 1*f[ih-2]
        if k == 3:
            f[il] = 4*f[il+1] - 6*f[il+2] + 4*f[il+3] - f[il+4]
            f[ih] = 4*f[ih-1] - 6*f[ih-2] + 4*f[ih-3] - f[ih-4]
        if k == 5:
            f[il] = 6*f[il+1]-15*f[il+2]+20*f[il+3]-15*f[il+4]+6*f[il+5]-f[il+6]  # noqa
            f[ih] = 6*f[ih-1]-15*f[ih-2]+20*f[ih-3]-15*f[ih-4]+6*f[ih-5]-f[ih-6]  # noqa
        if k == 7:
            f[il] = 8*f[il+1]-28*f[il+2]+56*f[il+3]-70*f[il+4]+56*f[il+5]-28*f[il+6]+8*f[il+7]-f[il+8]  # noqa
            f[ih] = 8*f[ih-1]-28*f[ih-2]+56*f[ih-3]-70*f[ih-4]+56*f[ih-5]-28*f[ih-6]+8*f[ih-7]-f[ih-8]  # noqa
        if k == 9:
            f[il] = 10*f[il+1]-45*f[il+2]+120*f[il+3]-210*f[il+4]+252*f[il+5]-210*f[il+6]+120*f[il+7]-45*f[il+8]+10*f[il+9]-f[il+10]  # noqa
            f[ih] = 10*f[ih-1]-45*f[ih-2]+120*f[ih-3]-210*f[ih-4]+252*f[ih-5]-210*f[ih-6]+120*f[ih-7]-45*f[ih-8]+10*f[ih-9]-f[ih-10]  # noqa


@nb.njit(
    nb.void(
        nb.float64[:, :],
        nb.int64,
        nb.int64,
    ),
    fastmath=True,
)
def _extrapolate1d_y(f, k, o):
    for ix in range(o):
        il = o-ix-1
        ih = f.shape[1]-(o-ix)
        if k == 1:
            f[:, il] = 2*f[:, il+1] - 1*f[:, il+2]
            f[:, ih] = 2*f[:, ih-1] - 1*f[:, ih-2]
        if k == 3:
            f[:, il] = 4*f[:, il+1] - 6*f[:, il+2] + 4*f[:, il+3] - f[:, il+4]
            f[:, ih] = 4*f[:, ih-1] - 6*f[:, ih-2] + 4*f[:, ih-3] - f[:, ih-4]
        if k == 5:
            f[:, il] = 6*f[:, il+1]-15*f[:, il+2]+20*f[:, il+3]-15*f[:, il+4]+6*f[:, il+5]-f[:, il+6]  # noqa
            f[:, ih] = 6*f[:, ih-1]-15*f[:, ih-2]+20*f[:, ih-3]-15*f[:, ih-4]+6*f[:, ih-5]-f[:, ih-6]  # noqa
        if k == 7:
            f[:, il] = 8*f[:, il+1]-28*f[:, il+2]+56*f[:, il+3]-70*f[:, il+4]+56*f[:, il+5]-28*f[:, il+6]+8*f[:, il+7]-f[:, il+8]  # noqa
            f[:, ih] = 8*f[:, ih-1]-28*f[:, ih-2]+56*f[:, ih-3]-70*f[:, ih-4]+56*f[:, ih-5]-28*f[:, ih-6]+8*f[:, ih-7]-f[:, ih-8]  # noqa
        if k == 9:
            f[:, il] = 10*f[:, il+1]-45*f[:, il+2]+120*f[:, il+3]-210*f[:, il+4]+252*f[:, il+5]-210*f[:, il+6]+120*f[:, il+7]-45*f[:, il+8]+10*f[:, il+9]-f[:, il+10]  # noqa
            f[:, ih] = 10*f[:, ih-1]-45*f[:, ih-2]+120*f[:, ih-3]-210*f[:, ih-4]+252*f[:, ih-5]-210*f[:, ih-6]+120*f[:, ih-7]-45*f[:, ih-8]+10*f[:, ih-9]-f[:, ih-10]  # noqa


@nb.njit(
    nb.void(
        nb.float64[:, :],
        nb.float64[:, :],
        nb.int64,
        nb.int64,
    ),
    fastmath=True,
)
def __fill2(f, fb, ox, oy):
    nx = f.shape[0]
    ny = f.shape[1]
    for i in nb.prange(nx):
        for j in range(ny):
            fb[i+ox, j+oy] = f[i, j]


@nb.njit(
    nb.void(
        nb.float64[:, :],
        nb.float64[:, :],
        nb.int64,
        nb.int64,
    ),
    fastmath=True,
)
def _fill2(f, fb, ox, oy):
    nx = f.shape[0]
    ny = f.shape[1]
    if nx*ny < 100000:
        fb[ox:ox+nx, oy:oy+ny] = f
    else:
        __fill2(f, fb, ox, oy)


@nb.njit(
    nb.types.Tuple((nb.float64[:, :], nb.int64[:]))(
        nb.float64[:, :],
        nb.int64,
        nb.boolean[:],
        nb.boolean[:],
        nb.int64[:],
    ),
    fastmath=True,
)
def _extrapolate2d(f, k, p, c, e):
    padx = (not p[0]) and c[0]
    pady = (not p[1]) and c[1]
    if padx or pady:
        ox = (k//2)+e[0] if padx else 0
        oy = (k//2)+e[1] if pady else 0
        fb = np.zeros(
            (f.shape[0]+2*ox, f.shape[1]+2*oy),
            dtype=f.dtype
        )
        _fill2(f, fb, ox, oy)
        if padx:
            _extrapolate1d_x(fb, k, ox)
        if pady:
            _extrapolate1d_y(fb, k, oy)
        return fb, np.array([ox, oy])
    else:
        return f, np.array([0, 0])


spec_2d = [
    ('a', nb.float64[:]),
    ('b', nb.float64[:]),
    ('h', nb.float64[:]),
    ('f', nb.float64[:, :]),
    ('k', nb.int64),
    ('p', nb.boolean[:]),
    ('c', nb.boolean[:]),
    ('e', nb.int64[:]),
    ('n', nb.int64[:]),
    ('_f', nb.float64[:, :]),
    ('_o', nb.int64[:]),
    ('lb', nb.float64[:]),
    ('ub', nb.float64[:]),
    ('xout', nb.float64[:]),
    ('yout', nb.float64[:]),
    ('out', nb.float64[:]),
]


@nb.experimental.jitclass(spec_2d)
class nb_interp2d(object):

    def __init__(
        self,
        a,
        b,
        h,
        f,
        k,
        p=np.array([False]*2),
        c=np.array([True]*2),
        e=np.array([0]*2),
    ):
        """
        a, b: the lower and upper bounds of the interpolation region
        h:    the grid-spacing at which f is given
        f:    data to be interpolated
        k:    order of local taylor expansions (int, 1, 3, or 5)
        p:    whether the dimension is taken to be periodic
        c:    whether the array should be padded to allow accurate close eval
        e:    extrapolation distance, how far to allow extrap, in units of h
                (needs to be an integer)

        See the documentation for interp1d
        this function is the same, except that a, b, h, p, c, and e
        should be lists or tuples of length 2 giving the values for each
        dimension
        the function behaves as in the 1d case, except that of course padding
        is required if padding is requested in any dimension
        """

        self.a = a
        self.b = b
        self.h = h
        self.f = f
        self.k = k
        self.p = p
        self.c = c
        self.e = e
        self.n = np.array(f.shape)
        _f, _o = _extrapolate2d(f, k, p, c, e)
        self._f = _f
        self._o = _o
        lb, ub = _compute_bounds(a, b, h, p, c, e, k)
        self.lb = lb
        self.ub = ub

    def eval(self, xout, yout):
        """
        Interpolate to xout
        For 1-D interpolation, xout must be a float
            or a ndarray of floats
        """

        m = int(np.prod(np.array(xout.shape)))

        self.xout = xout
        self.yout = yout
        self.out = np.empty(m, dtype=self.f.dtype)

        if self.k == 1:
            self.interp2d_k1()
        elif self.k == 3:
            self.interp2d_k3()
        elif self.k == 5:
            self.interp2d_k5()
        elif self.k == 7:
            self.interp2d_k7()
        elif self.k == 9:
            self.interp2d_k9()
        else:
            raise ValueError(f"No interpolation for k={self.k}")

        return self.out

    def interp2d_k1(self):
        _interp2d_k1(
            self._f,
            self.xout, self.yout, self.out,
            self.a,
            self.h,
            self.n,
            self.p,
            self._o,
            self.lb, self.ub
        )

    def interp2d_k3(self):
        _interp2d_k3(
            self._f,
            self.xout, self.yout, self.out,
            self.a,
            self.h,
            self.n,
            self.p,
            self._o,
            self.lb, self.ub
        )

    def interp2d_k5(self):
        _interp2d_k5(
            self._f,
            self.xout, self.yout, self.out,
            self.a,
            self.h,
            self.n,
            self.p,
            self._o,
            self.lb, self.ub
        )

    def interp2d_k7(self):
        _interp2d_k7(
            self._f,
            self.xout, self.yout, self.out,
            self.a,
            self.h,
            self.n,
            self.p,
            self._o,
            self.lb, self.ub
        )

    def interp2d_k9(self):
        _interp2d_k9(
            self._f,
            self.xout, self.yout, self.out,
            self.a,
            self.h,
            self.n,
            self.p,
            self._o,
            self.lb, self.ub
        )

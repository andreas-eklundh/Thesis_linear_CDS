import numpy as np
from numba import njit

#### Fir cascading.
@njit
def lin_interp(x, xp, fp):
    n = len(xp)
    if x <= xp[0]:
        return fp[0]
    if x >= xp[n-1]:
        return fp[n-1]

    lo = 0
    hi = n - 1
    # binary search
    while hi - lo > 1:
        mid = (hi + lo) // 2
        if xp[mid] > x:
            hi = mid
        else:
            lo = mid

    x0, x1 = xp[lo], xp[hi]
    f0, f1 = fp[lo], fp[hi]
    return f0 + (x - x0) * (f1 - f0) / (x1 - x0)


@njit
def laplace_from_grid(tau_vec, lam, tau_grid, alpha_grid, beta_grid):
    out = np.empty(len(tau_vec))
    Xdim = beta_grid.shape[1]

    for i in range(len(tau_vec)):
        tau = tau_vec[i]
        # interpolate alpha(tau)
        a = lin_interp(tau, tau_grid, alpha_grid)

        # interpolate each beta_k(tau)
        b = 0.0
        for k in range(Xdim):
            b += lin_interp(tau, tau_grid, beta_grid[:, k]) * lam[k]

        out[i] = np.exp(a + b)

    return out


@njit
def deriv_from_grid(tau_vec, lam, tau_grid, alpha_x_grid, beta_x_grid):
    out = np.empty(len(tau_vec))
    Xdim = beta_x_grid.shape[1]

    for i in range(len(tau_vec)):
        tau = tau_vec[i]
        a = lin_interp(tau, tau_grid, alpha_x_grid)

        b = 0.0
        for k in range(Xdim):
            b += lin_interp(tau, tau_grid, beta_x_grid[:, k]) * lam[k]

        out[i] = a + b

    return out


@njit
def calc_prot_numba(t, t0, t_mat, lam,
                    tau_grid, alpha_grid, beta_grid,
                    alpha_x_grid, beta_x_grid,
                    r, delta):
    tau_start = t0 - t
    tau_end   = t_mat - t
    if tau_end <= tau_start:
        return 0.0

    N = 300  # integration resolution
    dt = (tau_end - tau_start) / (N - 1)

    total = 0.0
    tau = tau_start

    for _ in range(N-1):
        tm = tau + 0.5*dt

        d  = np.exp(-r * tm)
        df = deriv_from_grid(np.array([tm]), lam,
                             tau_grid, alpha_x_grid, beta_x_grid)[0]
        lt = laplace_from_grid(np.array([tm]), lam,
                               tau_grid, alpha_grid, beta_grid)[0]

        total += d * df * lt * dt
        tau += dt

    return (1.0 - delta) * total


@njit
def calc_coupon_numba(t, t0, t_mat, lam,
                      tau_grid, alpha_grid, beta_grid,
                      r, tenor):
    # coupon dates
    # Python arange is not numba-friendly → build manually
    n_coupons = int(np.floor((t_mat - t0) / tenor)) + 1
    t_grid = np.empty(n_coupons)
    for i in range(n_coupons):
        t_grid[i] = t0 + i * tenor

    if len(t_grid) < 2:
        return 0.0

    total = 0.0
    for j in range(1, len(t_grid)):
        a = t_grid[j-1]
        b = t_grid[j]

        tau = b - t
        disc = np.exp(-r * tau)
        lt = laplace_from_grid(np.array([tau]), lam,
                               tau_grid, alpha_grid, beta_grid)[0]

        total += (b - a) * disc * lt

    return total


@njit
def calc_accrual_numba(t, t0, t_mat, lam,
                       tau_grid, alpha_grid, beta_grid,
                       alpha_x_grid, beta_x_grid,
                       r, tenor):
    n_coupons = int(np.floor((t_mat - t0) / tenor)) + 1
    t_grid = np.empty(n_coupons)
    for i in range(n_coupons):
        t_grid[i] = t0 + i * tenor

    if len(t_grid) < 2:
        return 0.0

    total = 0.0
    N = 40  # per coupon integration resolution

    for j in range(1, len(t_grid)):
        a = t_grid[j-1]
        b = t_grid[j]

        tau_start = a - t
        dt = (b - a) / (N - 1)

        tau = tau_start
        for _ in range(N-1):
            tm = tau + 0.5 * dt
            u  = tm + t
            dfac = u - a

            disc = np.exp(-r * tm)
            deriv = deriv_from_grid(np.array([tm]), lam,
                                    tau_grid, alpha_x_grid, beta_x_grid)[0]
            lt = laplace_from_grid(np.array([tm]), lam,
                                   tau_grid, alpha_grid, beta_grid)[0]

            total += disc * dfac * deriv * lt * dt
            tau += dt

    return total


@njit
def calc_CDS_numba(t, t0, t_mat, lam,
                   tau_grid, alpha_grid, beta_grid,
                   alpha_x_grid, beta_x_grid,
                   r, delta, tenor):

    prot = calc_prot_numba(t, t0, t_mat, lam,
                           tau_grid, alpha_grid, beta_grid,
                           alpha_x_grid, beta_x_grid,
                           r, delta)

    cpn  = calc_coupon_numba(t, t0, t_mat, lam,
                             tau_grid, alpha_grid, beta_grid,
                             r, tenor)

    accr = calc_accrual_numba(t, t0, t_mat, lam,
                              tau_grid, alpha_grid, beta_grid,
                              alpha_x_grid, beta_x_grid,
                              r, tenor)

    denom = cpn + accr
    if denom == 0:
        return 0.0

    return prot / denom
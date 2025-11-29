import numpy as np
from numba import njit

# --------------------------------------------------------------------
# CIR helpers (as before)
# --------------------------------------------------------------------
@njit
def unpack_params(params, X_dim):
    kappa = params[:X_dim]
    theta = params[X_dim:2 * X_dim]
    sigma = params[2 * X_dim:3 * X_dim]
    kappa_p = params[3 * X_dim:4 * X_dim]
    theta_p = params[4 * X_dim:5 * X_dim]
    sigma_err = np.array([params[-1]])
    return kappa, theta, sigma, kappa_p, theta_p, sigma_err


@njit
def cir_solution(params, x0, T_arr, X_dim):
    kappa, theta, sigma1, kappa_p, theta_p, sigma_err = unpack_params(params, X_dim)
    rho = 1.0
    gamma = np.sqrt(kappa ** 2 + 2.0 * sigma1 ** 2 * rho)

    alpha = 0.0
    beta = np.zeros((X_dim))

    for i in range(X_dim):
        exp_term = np.exp(gamma[i] * T_arr)
        beta_nom = (-2.0 * rho * (exp_term - 1.0)
                    + x0[i] * exp_term * (gamma[i] - kappa[i])
                    + x0[i] * (gamma[i] + kappa[i]))
        beta_denom = (2.0 * gamma[i]
                      + (gamma[i] + kappa[i] - x0[i] * sigma1[i] ** 2)
                      * (exp_term - 1.0))
        tmp = beta_nom / beta_denom
        beta[i] = tmp    
        alpha_log_nom = 2.0 * gamma[i] * np.exp((gamma[i] + kappa[i]) * T_arr / 2.0)
        alpha_log_denom = (2.0 * gamma[i]
                           + (gamma[i] + kappa[i] - x0[i] * sigma1[i] ** 2)
                           * (exp_term - 1.0))
        alpha += (2.0 * kappa[i] * theta[i]
                  * np.log(alpha_log_nom / alpha_log_denom)
                  / sigma1[i] ** 2)

    return alpha, beta


@njit
def cir_derivatives(params, x0, T_arr, X_dim):
    kappa, theta, sigma1, kappa_p, theta_p, sigma_err = unpack_params(params, X_dim)
    rho = 1.0
    gamma = np.sqrt(kappa ** 2 + 2.0 * sigma1 ** 2 * rho)

    alpha_x = 0.0
    beta_x = np.zeros((X_dim))

    for i in range(X_dim):
        exp_term = np.exp(gamma[i] * T_arr)
        denom = (2.0 * gamma[i]
                 + (gamma[i] + kappa[i] - x0[i] * sigma1[i] ** 2)
                 * (exp_term - 1.0))

        bterm1 = - (2.0 * rho * (exp_term - 1.0) ** 2 * sigma1[i] ** 2) / (denom ** 2)
        bterm2 = (exp_term * (gamma[i] - kappa[i]) + (gamma[i] + kappa[i])) / denom
        bterm3 = (x0[i] * (exp_term * (gamma[i] - kappa[i]) + (gamma[i] + kappa[i]))
                  * sigma1[i] ** 2 * (exp_term - 1.0) / (denom ** 2))
        beta_x[i] = bterm1 + bterm2 + bterm3
        alpha_x += 2.0 * kappa[i] * theta[i] * (exp_term - 1.0) / denom

    return alpha_x, beta_x


@njit
def laplace_transform(params, lambda_t, x0, T_arr, X_dim):
    alpha, beta = cir_solution(params, x0, T_arr, X_dim)
    return np.exp(alpha + np.dot(beta, lambda_t))


# --------------------------------------------------------------------
# Trapezoidal integration
# --------------------------------------------------------------------
@njit
def trapezoidal_rule(integrand, t_start, t_end, n_steps, *args):
    dt = (t_end - t_start) / n_steps
    total = 0.0
    t_prev = t_start
    f_prev = integrand(t_prev, *args)
    for i in range(1, n_steps + 1):
        t_curr = t_start + i * dt
        f_curr = integrand(t_curr, *args)
        total += 0.5 * (f_curr + f_prev) * dt
        t_prev = t_curr
        f_prev = f_curr
    return total


# --------------------------------------------------------------------
# Integrands and Legs
# --------------------------------------------------------------------

@njit
def _get_default_grid( u, t_grid):
    if u <= t_grid[0]:
        return 0.0
    if u >= t_grid[-1]:
        return t_grid[-1] - t_grid[-2]  # last interval length
    idx = np.searchsorted(t_grid, u) - 1
    return u - t_grid[idx]

@njit
def accrual_integrand(u, params, t, lambda_t, r, X_dim,t_grid):
    tau = u - t
    alpha_x, beta_x = cir_derivatives(params, np.zeros(X_dim), tau, X_dim)
    L = laplace_transform(params, lambda_t, np.zeros(X_dim), tau, X_dim)
    return np.exp(-r * tau) * (alpha_x + np.dot(beta_x, lambda_t)) * L *(_get_default_grid(u,t_grid))


@njit
def protection_integrand(u, params, t, lambda_t, r, delta, X_dim,t_grid):
    tau = u - t
    alpha_x, beta_x = cir_derivatives(params, np.zeros(X_dim), tau, X_dim)
    L = laplace_transform(params, lambda_t, np.zeros(X_dim), tau, X_dim)
    return (1.0 - delta) * np.exp(-r * tau) * (alpha_x + np.dot(beta_x, lambda_t)) * L


# --------------------------------------------------------------------
# Full legs (same structure as your cir_n version)
# --------------------------------------------------------------------
@njit
def calc_coupon_leg(params, t,t0, t_mat, lambda_t, r, tenor, X_dim,n_steps):
    I = np.zeros(1)
    t_grid = np.arange(t0, t_mat + 1e-12, tenor)
    for t_idx in range(1, len(t_grid)):
        expectation =  laplace_transform(params, lambda_t, np.zeros(X_dim), t_grid[t_idx] - t, X_dim)
        I += (t_grid[t_idx]-t_grid[t_idx-1]) * np.exp(-r * (t_grid[t_idx] - t)) * expectation
    return I

@njit
def calc_accrual_leg(params, t,t0, t_mat, lambda_t, r, tenor, X_dim,n_steps):
    t_grid = np.arange(t0, t_mat + 1e-12, tenor)

    integral = trapezoidal_rule(accrual_integrand, t0, t_mat, n_steps, params, t, lambda_t, r, X_dim,t_grid)
    return np.asarray(integral).flatten()

@njit
def calc_protection_leg(params, t,t0, t_mat, lambda_t, r, delta, tenor, X_dim,n_steps):
    t_grid = np.arange(t0, t_mat + 1e-12, tenor)

    integral = trapezoidal_rule(protection_integrand, t0, t_mat, n_steps, params, t, lambda_t, r, delta, X_dim,t_grid)
    return np.asarray(integral).flatten()

# --------------------------------------------------------------------
# Combine to CDS
# --------------------------------------------------------------------
@njit
def calc_cds(params, t, t_mat, lambda_t, t0, r, delta, tenor, X_dim):
    n_steps = 500
    prot_val = calc_protection_leg(params, t,t0, t_mat, lambda_t, r, delta, tenor, X_dim,n_steps)
    I1 = calc_coupon_leg(params, t,t0, t_mat, lambda_t, r, tenor, X_dim,n_steps)
    I2 = calc_accrual_leg(params, t,t0, t_mat, lambda_t, r, tenor, X_dim,n_steps)
    return prot_val / (I1 + I2)

### Calculate several spreads.

@njit
def cals_cds_several(X,params, t, t_mat_grid,t0=None):
    # If no t0 provided, assume at inception
    if t0 == None:
        t0 = t
    result = np.zeros(t_mat_grid.shape[0], dtype=np.float64)

    for i in range(t_mat_grid.shape[0]):
        # Pass a scalar from the array X
        # Make sure that t grid is of size 1 due to logic.
        t_mat = np.array([t_mat_grid[i]])
        result[i] = calc_cds(params,t, t_mat, X,t0)[0] # A
    return result



#### Fir cascading.

# ============================================================
# 1. NUMBA-FRIENDLY LINEAR INTERPOLATOR
# ============================================================
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


# ============================================================
# 2. NUMBA FUNCTIONS FOR AFFINE LAPLACE EVALUATION
# ============================================================
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


# ============================================================
# 3. NUMBA-PROTECTION LEG
# ============================================================
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


# ============================================================
# 4. NUMBA-COUPON LEG
# ============================================================
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


# ============================================================
# 5. NUMBA-ACCRUAL LEG
# ============================================================
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


# ============================================================
# 6. NUMBA-CDS PRICE
# ============================================================
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
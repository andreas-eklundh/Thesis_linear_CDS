import numpy as np
import math
from scipy.optimize import minimize, NonlinearConstraint,LinearConstraint , Bounds
from scipy.stats import norm, ncx2, gamma, expon, uniform
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from itertools import combinations_with_replacement, product
import re
from numpy.polynomial.legendre import legval,Legendre
from Models.moments import MultivariatePolynomialGenerator as mpg
import sympy as sp
import time

from scipy.optimize import lsq_linear
from numba import njit, float64, int64
from numba.experimental import jitclass
from scipy.integrate import quad
import copy
from scipy.linalg import sqrtm
from scipy.optimize import brentq
from scipy.optimize import differential_evolution
from types import SimpleNamespace


#### PURPOSE: COPY OF LHC_SINGLE THAT CAN OPTIMIZE WITH GAMMA1.

spec = [
    ('a', float64[:,:]),        # row/column -> flatten to 1D
    ('c', float64[:,:]),
    ('gamma', float64[:,:]),    # row vector -> 1D
    ('b', float64[:,:]),
    ('beta', float64[:,:]),
    ('A', float64[:,:]),
    ('A_star', float64[:,:]),
    ('A_star_inv', float64[:,:]),
    ('id_mat', float64[:,:]),
    ('r', float64),
    ('m', int64),
    ('Y_dim', int64),
    ('delta', float64),
    ('tenor', float64),
    ('sum_Z',float64[:]),
    ('sum_D',float64[:]),

]

# -------- Matrix exponential approx (safe for numba) -------- #
@njit
def frobenius_norm(mat):
    n, m = mat.shape
    s = 0.0
    for i in range(n):
        for j in range(m):
            s += mat[i, j] * mat[i, j]
    return np.sqrt(s)

@njit
def mat_exp_approx(A, dt, tol=1e-9):
    n = A.shape[0]
    I = np.eye(n)
    Adt = A * dt

    mat_expo = I.copy()
    term = I.copy()

    # We use a fixed upper limit to prevent infinite loops in cases of non-convergence
    limit = 70 # should be more than sufficient.

    for i in range(1, limit + 1):
        # Calculate the next term
        term = term @ Adt / i   # matrix multiply works in numba

        # Check convergence via Frobenius norm
        if frobenius_norm(term) < tol:
            break

        # Add the new term to the running sum
        mat_expo += term

    return mat_expo

# Rebuild dynamics and return the LHCStruct
@njit
def rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor,lambda_i=None):
    m = theta.shape[0]
    if lambda_i is None:
        lambda_i = np.zeros(m)
    # b: shape (m, Y_dim)
    b = np.zeros((m, Y_dim))
    b[m-1, 0] = theta[-1] * kappa[-1]

    # beta: shape (m, m)
    beta = np.zeros((m, m))
    for i in range(m):
        beta[i, i] = - (kappa[i])
        if i + 1 < m:
            beta[i, i+1] = kappa[i] * theta[i]

    # gamma: row vector, shape (1, m)
    gamma = np.zeros((1, m))
    gamma[0, 0] = - gamma1

    # c: shape (Y_dim, Y_dim)
    c = np.zeros((Y_dim, Y_dim))

    # A: shape (Y_dim + m, Y_dim + m)
    A = np.zeros((Y_dim + m, Y_dim + m))
    A[:Y_dim, :Y_dim] = c
    A[:Y_dim, Y_dim:] = gamma
    A[Y_dim:, :Y_dim] = b
    A[Y_dim:, Y_dim:] = beta

    A = np.ascontiguousarray(A)
    b = np.ascontiguousarray(b)
    beta = np.ascontiguousarray(beta)

    # Identity, A_star, and inverse
    id_mat = np.eye(Y_dim + m)
    A_star = A - r * id_mat

    # det_A_star = np.linalg.det(A_star)
    # if (np.isnan(det_A_star))|(np.abs(det_A_star)<1e-12) :
    #     A_star_inv = np.linalg.pinv(A_star)
    # else:
    A_star_inv = np.linalg.inv(A_star)

    # a vector (assume ones for simplicity)
    a = np.ones((Y_dim,1))


    lhc_tuple = (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor)

    return lhc_tuple


# -------- psi functions rewritten for numba -------- #
@njit
def psi_Z(lhc, t, t_M):
    (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor) = lhc
    dt = t_M - t
    mat_exp = mat_exp_approx(A, dt)
    a0 = np.zeros(Y_dim + m)
    a0[:Y_dim] = a.ravel()
    
    return np.exp(-r * dt) * (a0 @ mat_exp).ravel()

@njit
def psi_D(lhc, t, t_M):
    (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor) = lhc
    dt = t_M - t
    mat_exp = mat_exp_approx(A_star, dt)
    # build [c | gamma]
    c_gamma = np.zeros((Y_dim, Y_dim + m))
    c_gamma[:, :Y_dim] = c
    c_gamma[:, Y_dim:] = gamma
    tmp = mat_exp - id_mat
    a_row = a.ravel()
    return -(a_row @ c_gamma @ (A_star_inv @ tmp)).ravel()

@njit
def psi_D_star(lhc, t, t_M):
    (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor) = lhc
    dt = t_M - t
    mat_exp = mat_exp_approx(A_star, dt)
    c_gamma = np.zeros((Y_dim, Y_dim + m))
    c_gamma[:, :Y_dim] = c
    c_gamma[:, Y_dim:] = gamma
    term1 = dt * (A_star_inv @ mat_exp)
    term2 = A_star_inv @ ((id_mat * t - A_star_inv) @ (mat_exp - id_mat))
    a_row = a.ravel()
    return -(a_row @ c_gamma @ (term1 + term2)).ravel()

@njit
def psi_prot(lhc, t, t0, t_M):
    (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor) = lhc
    return (1.0 - delta) * (psi_D(lhc, t, t_M) - psi_D(lhc, t, t0))

@njit
def psi_prem(lhc, t, t0, t_M):
    (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor) = lhc
    sum_Z = np.zeros(Y_dim + m)
    sum_D = np.zeros(Y_dim + m)
    t_grid = np.arange(t0, t_M + 1e-12, tenor)
    dt = t_grid[1] - t_grid[0]
    for j in range(1, len(t_grid)):
        sum_Z += dt * psi_Z(lhc, t, t_grid[j])
        if j < len(t_grid) - 1:
            sum_D += dt * psi_D(lhc, t, t_grid[j])
    coupon_leg = sum_Z
    accrual_default =  psi_D_star(lhc, t, t_M) - psi_D_star(lhc, t, t0)
    accrual_prev = t_grid[-2] * psi_D(lhc, t, t_M) - sum_D - t0 * psi_D(lhc, t, t0)
    return (coupon_leg+accrual_default - accrual_prev).flatten()

@njit 
def psi_prem_fast(lhc,t,t0,t_M_list,coupon_leg_mat):
    accrual_default = np.zeros_like(coupon_leg_mat)
    accrual_prev = np.zeros_like(coupon_leg_mat)
    for i,t_m in enumerate(t_M_list):
        accrual_default[i,:] = (psi_D_star(lhc, t, t_m) - psi_D_star(lhc, t, t0)).flatten()
        # hardcoded tenor
        accrual_prev[i,:] = (t_m - 0.25) * psi_D(lhc, t, t_m) - t0 * psi_D(lhc, t, t0)

    return coupon_leg_mat+accrual_default-accrual_prev

@njit
def psi_prem_pre(lhc, t, t0, t_M):
    '''
    Compute all but accrual part.
    '''
    (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor) = lhc
    sum_Z = np.zeros(Y_dim + m)
    sum_D = np.zeros(Y_dim + m)
    t_grid = np.arange(t0, t_M + 1e-12, tenor)
    dt = t_grid[1] - t_grid[0]
    for j in range(1, len(t_grid)):
        sum_Z += dt * psi_Z(lhc, t, t_grid[j])
        if j < len(t_grid) - 1:
            sum_D += dt * psi_D(lhc, t, t_grid[j])
    coupon_leg = sum_Z
    # accrual_default =  psi_D_star(lhc, t, t_M) - psi_D_star(lhc, t, t0)
    return (coupon_leg + sum_D).flatten()


@njit
def psi_cds(lhc, t, t0, t_M, k):
    return psi_prot(lhc, t, t0, t_M) - k * psi_prem(lhc, t, t0, t_M)

@njit
def get_CDS_Model(t_obs, t0, t_mat_grid, state_vec, lhc,calc_efficiently=False):
    n_mat, n_obs = t_mat_grid.shape
    CDS = np.ones((n_mat, n_obs))

    # If in calibration setup calibrate fast by precomputing.
    if calc_efficiently:
        # Compute Psi's for mats. Only prot is a Protection leg is time homogene.
        psi_prot_mat = np.zeros((n_mat, state_vec.shape[0]))
        psi_prem_mat = np.zeros((n_mat, state_vec.shape[0]))
        psi_prem_pre_mat = np.zeros((n_mat, state_vec.shape[0]))
    
        for i in range(n_mat):
            # Pass a scalar from the array X
            psi_prot_mat[i,:] = psi_prot(lhc,t_obs[0],t0[0],t_mat_grid[i,0])
            psi_prem_pre_mat[i,:] = psi_prem_pre(lhc,t_obs[0],t0[0],t_mat_grid[i,0])

        psi_prot_mat = np.ascontiguousarray(psi_prot_mat)
        # Find Z,X,Y
        for time_idx in range(0, n_obs):
            # Compute psi's
            psi_prem_mat = psi_prem_fast(lhc,t_obs[time_idx],t_obs[time_idx],
                                        t_mat_grid[:,time_idx].flatten(),psi_prem_pre_mat)
            psi_prem_mat = np.ascontiguousarray(psi_prem_mat)
            st = np.ascontiguousarray(state_vec[:, i])
            CDS[:, i] = psi_prot_mat @ st / psi_prem_mat @ st
    else:
        for mat_idx in range(n_mat):
            for i in range(n_obs):
                prot = psi_prot(lhc, t_obs[i], t0[i], t_mat_grid[mat_idx, i])
                prem = psi_prem(lhc, t_obs[i], t0[i], t_mat_grid[mat_idx, i])
                st = np.ascontiguousarray(state_vec[:, i])
                CDS[mat_idx, i] = np.dot(prot, st) / np.dot(prem, st)
    return CDS

@njit
def cds_fun(lhc, chi, t,t0, t_mat_grid):
    result = np.zeros(t_mat_grid.shape[0], dtype=np.float64)
    for i in range(t_mat_grid.shape[0]):
        # Pass a scalar from the array X
        prem = psi_prem(lhc,t,t0,t_mat_grid[i])
        prot = psi_prot(lhc,t,t0,t_mat_grid[i])
        result[i] = np.dot(prot, chi) / np.dot(prem, chi)

    return result

@njit
def cds_deriv(lhc, chi, t,t0, t_mat_grid):
    (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor) = lhc
    result = np.zeros((t_mat_grid.shape[0],m), dtype=np.float64)
    for i in range(t_mat_grid.shape[0]):
        # Pass a scalar from the array X
        prem = psi_prem(lhc,t,t0,t_mat_grid[i])
        prot = psi_prot(lhc,t,t0,t_mat_grid[i])
        term1 = prot[1:] / np.dot(prem, chi)
        term2 = np.dot(prot, chi) / np.dot(prem, chi)**2 * prem[1:]
        result[i,:] = term1 - term2

    return result


## Wrappers for the above to be used in Kalman filter. Avoids t_m + t_m*m comps per loop.
@njit
def cds_fun_mats(prem,prot,chi):
    prod_prem = prem @ chi # Array of len t_Mats
    prod_prot = prot @ chi
    # Outerprod the way out

    return prod_prot / prod_prem

@njit
def cds_deriv_mats(prem,prot,chi):
    prod_prem = prem @ chi # Array of len t_Mats
    prod_prot = prot @ chi
    # Outerprod the way out
    term1 = prot[:, 1:] / prod_prem[:, None]
    term2 = (prod_prot / prod_prem**2)[:, None] * prem[:, 1:]
    result = term1 - term2

    return result



@njit
def cds_fun_lin(lhc, chi_m1, t,t0, t_mat_grid):
    (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor) = lhc
    chi_m1 = np.append([1],chi_m1)
    result = np.zeros((t_mat_grid.shape[0],int(Y_dim+ m)), dtype=np.float64)
    for i in range(t_mat_grid.shape[0]):
        # Pass a scalar from the array X
        prem = psi_prem(lhc,t,t0,t_mat_grid[i])
        prot = psi_prot(lhc,t,t0,t_mat_grid[i])
        result[i,:] = prot / np.dot(prem, chi_m1)

    return result

## Try to run on value of CDS instead.
@njit
def cds_value(lhc, t,t0, t_mat_grid,CDS_grid):
    (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor) = lhc
    result = np.zeros((t_mat_grid.shape[0],int(Y_dim+ m)), dtype=np.float64)
    for i in range(t_mat_grid.shape[0]):
        # Pass a scalar from the array X
        prem = psi_prem(lhc,t,t0,t_mat_grid[i])
        prot = psi_prot(lhc,t,t0,t_mat_grid[i])
        result[i,:] = prot - CDS_grid[i] * prem

    return result




# standard kalman with previous in denom.
@njit
def update_step_cds(X_pred, P_pred, h,h_p, R_k, prem,prot, CDS_k):
    pred_Xn = np.append([1],X_pred) # Add one for computations of cds spread and derivative
    # Convert to contingous arrays in case.
    P_pred = np.ascontiguousarray(P_pred)

    # Step 3: Mean prediction, covariance, Kalman Gain etc.
    # Iterated Kalman Filter. For I_max=1 standard.
    I_max = 1
    for i in range(0,I_max):
        # mu_k = h(lhc, pred_Xn, t_obs,t0, t_mats)

        # H_x = h_p(lhc, pred_Xn, t_obs,t0, t_mats)
        mu_k = h(prem,prot,pred_Xn)

        H_x = h_p(prem,prot,pred_Xn)
        H_x = np.ascontiguousarray(H_x)

        # covariance
        S_k = H_x @ P_pred @ H_x.T + R_k
        S_k = np.ascontiguousarray(S_k)

        # det_S = np.linalg.det(S_k)
        # if  (np.isnan(det_S)) | (np.abs(det_S) < 1e-12 ):
        #     S_k_inv = np.linalg.pinv(S_k)
        # else:
        S_k_inv = np.linalg.inv(S_k)
        S_k_inv = np.ascontiguousarray(S_k_inv)

        # Step 4: Compute Kalman Gain, filtered mean state, covariance.
        K_k = P_pred @ H_x.T @ S_k_inv
        # vn = (CDS_k - mu_k) # In Linear Approx instance
        vn = (CDS_k - mu_k - H_x @ (X_pred - pred_Xn[1:])) # First order approx for iterated.

        m_k = X_pred + K_k @ vn

        # Bump pred_Xn
        pred_Xn = np.append([1],m_k)

        # Increment
    P_k = P_pred - K_k @ S_k @ K_k.T

    return vn,S_k, m_k, P_k, S_k_inv


@njit
def build_P_params(params,params_q,lhc_p):
        (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor) = lhc_p
        kappa = params_q[:m]
        theta = params_q[m:2*m]
        gamma1 = params_q[-1]

        lambda_i = params[:m]
        kappa_p = kappa - lambda_i # This is under (i) in THM B.1.
        theta_p = (kappa*theta) / kappa_p
        # theta_p[-1] = theta_p[-1] + lambda_i[-1] / kappa_p[-1]
        # theta_p = theta_p + lambda_i / kappa_p

        sigma_i,sigma_err = params[m:2*m], params[-1]

        # Rebuild P parameters.
        lhc = rebuild_lhc_struct(kappa_p, theta_p, gamma1, r, Y_dim, delta, tenor, lambda_i=None)


        return lhc,lambda_i, sigma_i, sigma_err



@njit
def build_matrices(lhc,sigma_i,sigma_err,n_mat):
    (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor) = lhc
    A_trans = np.zeros((Y_dim + m, Y_dim + m))
    A_trans[:Y_dim, :Y_dim] =  c # np.ones(shape = lhc.c.shape)
    A_trans[:Y_dim, Y_dim:] = gamma
    A_trans[Y_dim:, :Y_dim] = b
    A_trans[Y_dim:, Y_dim:] = beta

    # Get covariance.
    sigma = np.zeros((Y_dim + m,m))
    sigma[1:,0:] = np.diag(sigma_i)

    cov_trans = sigma

    cov_meas =  np.identity(n = int(n_mat)) * sigma_err**2


    return A_trans,cov_trans,cov_meas



### Actual kalman filters.
@njit
def drift_term(Xn,lhc_p,Delta):
    (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor) = lhc_p
    Matrix = Xn + ((beta + np.identity(beta.shape[0])*(- gamma @ Xn))@ Xn)*Delta
    const = b.flatten() * Delta
    return Matrix + const

@njit
def drift_deriv_term(Xn, lhc_p, Delta):
    (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor) = lhc_p
    x = Xn.reshape(-1)
    g = - gamma.flatten()   # ensure 1D
    n = x.size

    s = g @ x                    # scalar g^T x
    outer_xg = np.outer(x, g)    # x g^T

    # Plus here as we are finding derivative wrt. Z and not Y.
    J = np.eye(n) + Delta * (beta + (s * np.eye(n) + outer_xg))
    return J




@njit
def get_states(lhc, t_obs, T_M_grid, CDS_obs,kappa, theta, gamma1):
    (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor) = lhc
    # RETHINK THIS A LOT. SEEMS LIKELY THAT THERE IS SOME SORT OF ERROR HERE.
    n_obs = len(t_obs)
    n_mat = T_M_grid.shape[0]

    # Define initial values
    # X0 = np.ones(shape=(m,)) *X0
    X = np.ones((m, n_obs))
    Y = np.ones((n_obs)) # Implicitly sets Y0
    # Time 0 values
    # Todo: Rewrite this to start off at mu. then transform.
    mu1 = solve_mu1(kappa, theta, gamma1,lambda_i=None)
    mu = compute_stationary(kappa, theta,m, gamma1,mu1,lambda_i=None)
    X[:, 0] = mu
    # Previous Z, starting guess
    Z = np.ones((m,n_obs))
    Y_prev = Y[0]
    X_prev = np.ascontiguousarray(X[:,0])
    Z_prev = np.ascontiguousarray(X[:,0] / Y[0])

    # Precompute all possible of the psifunctions. expecially prem leg has low hanging fruits.

    # Compute Psi's for mats. Only prot is a Protection leg is time homogene.
    psi_prot_mat = np.zeros((n_mat, m+1))
    psi_prem_mat = np.zeros((n_mat, m+1))
    psi_prem_pre_mat = np.zeros((n_mat, m+1))
 
    for i in range(n_mat):
        # Pass a scalar from the array X
        psi_prot_mat[i,:] = psi_prot(lhc,t_obs[0],t_obs[0],T_M_grid[i,0])
        psi_prem_pre_mat[i,:] = psi_prem_pre(lhc,t_obs[0],t_obs[0],T_M_grid[i,0])

    psi_prot_mat = np.ascontiguousarray(psi_prot_mat)


    ti = t_obs[1]
    ti_prev = t_obs[0]
    dt = ti-ti_prev
    # Find Z,X,Y
    for time_idx in range(0, n_obs):
        # Compute psi's
        psi_prem_mat = psi_prem_fast(lhc,t_obs[time_idx],t_obs[time_idx],
                                     T_M_grid[:,time_idx].flatten(),psi_prem_pre_mat)
        psi_prem_mat = np.ascontiguousarray(psi_prem_mat)
        psi_c = psi_prot_mat -  psi_prem_mat * CDS_obs[time_idx,None].T

        # Actual optimization
        A_big = np.empty(shape = (n_mat,m))
        y_big = np.empty(shape = (n_mat))
        # Weight matrix
        W = np.zeros(shape=(n_mat,n_mat))
        ti = t_obs[time_idx]
        # Build stacked vector
        one_Z = np.empty(shape = (1 + m))
        one_Z[0] = 1.0
        one_Z[1:] = Z_prev.flatten()

        d_k = psi_prem_mat @ one_Z
        A_big = - psi_c[:,1:]
        y_big =  psi_c[:,0]      # Note, y needs to be negative to formulate as WLS problem
        W = np.diag(1 / d_k**2 )     # Needs to be squared to match reg .
        A_big = np.ascontiguousarray(A_big)
        y_big =  np.ascontiguousarray(y_big)
        # Maybe not correct formulat (generalized inverse)
        if (m == 1) & (n_mat == 1):
            Z[:,time_idx] =  np.clip(y_big / A_big,0.0,1.0)
        else:
            # Only linear in states, NOT PARAMETers
            Z[:,time_idx] = np.clip(
                                    np.linalg.pinv(A_big.T @ W @ A_big) @ A_big.T @ W @ y_big,
                                    0.0,
                                    1.0
                                    )

        # Update Y and X
        if time_idx == 0:
            Y[time_idx]= 1
        else:
            Y[time_idx] = ((Y_prev + dt * np.dot(gamma.flatten(), X_prev.flatten())))

        X[:,time_idx] = Y[time_idx] * Z[:,time_idx]


        # Bump previous value
        Z_prev = np.ascontiguousarray(Z[:,time_idx])
        X_prev = np.ascontiguousarray(X[:,time_idx])
        Y_prev = Y[time_idx]


    return X,Y,Z

# Define f(mu1)
@njit
def f(mu1, kappa, theta,gamma1,m,lambda_i):
    prod = 1.0
    for j in range(m):
        prod *= kappa[j] * theta[j] / (mu1 * gamma1 - (kappa[j]-lambda_i[j]))
    return ((-1)**m) * prod - mu1

@njit
def solve_mu1(kappa, theta, gamma1, lambda_i):
    m = len(kappa)
    if lambda_i is None:
        lambda_i = np.zeros(m)

    # Bisection method parameters
    a = 1e-6
    b = 1.0 - 1e-6
    fa = f(a, kappa, theta,gamma1,m,lambda_i)
    fb = f(b, kappa, theta,gamma1,m,lambda_i)

    # Check for valid sign change
    if fa * fb > 0 or not np.isfinite(fa) or not np.isfinite(fb):
        # fallback: safe clipped mean value
        return max(min(0.5, 0.99), 0.01)

    # Bisection loop
    for _ in range(100):
        mid = 0.5 * (a + b)
        fm = f(mid, kappa, theta,gamma1,m,lambda_i)
        if abs(fm) < 1e-10 or (b - a) < 1e-10:
            return mid
        if fa * fm < 0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm
    return 0.5 * (a + b)

@njit
def compute_stationary(kappa, theta, m, gamma1, mu1,lambda_i):
    mu_process = np.zeros(m)
    if lambda_i is None:
        lambda_i = np.zeros(m)
    for i in range(m-1, -1, -1):
        sign = (-1)**(m - (i+1) + 1)
        prod = 1.0
        for j in range(i, m):
            prod *= kappa[j] * theta[j] / (mu1 * gamma1 - (kappa[j]-lambda_i[j]))
        mu_process[i] = sign * prod
    return mu_process

@njit
def calc_gamma1(kappa, theta, lambda_i=None):
    '''
    Lambda fct is only legacy. just plug in values for kappa_p,thetap when needed. 
    '''
    # Get stationary lambda1. 
    # stationary_spread = 5 / 10000
    # # NOTE: hard coded recovery rate. cant be bothered to expand at this point
    # lambda_bar1 = stationary_spread #/ (1-0.4)
    # m = len(kappa)
    # if lambda_i is None:
    #     lambda_i = np.zeros(m)
    # prod = 1.0
    # for j in range(m):
    #     prod *= kappa[j] * theta[j] / (lambda_bar1 - (kappa[j]-lambda_i[j]))
    # gamma1 = lambda_bar1 / (((-1)**m) * prod)

    # Solve only for admissible range
    gamma1 = kappa[0] / 2
    return np.array([gamma1])


# One kalman filter for optimizing and one for outputting.
@njit
def kalmanfilter_opt(params, t_obs,t0,T_M_grid,CDS_obs,lhc_p,lhc_q,X0):
    (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor) = lhc_q
    (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor) = lhc_p
    # Define the parameters already to be able to look over them
    params_q = params[:2*m+1]
    kappa, theta,gamma1 = params_q[:m],params_q[m:2*m],params_q[-1]
    params_p = params[2*m+1:]
    lambda_i,sigma,sigma_err = params_p[:m],params_p[m:2*m], params_p[-1]
    kappa_p = kappa - lambda_i
    theta_p = (kappa*theta) / kappa_p
    # theta_p[-1] = theta_p[-1] + lambda_i[-1] / kappa_p[-1]    # Get initial guesses.
    # theta_p = theta_p + lambda_i / kappa_p

    n_obs = CDS_obs.shape[0]
    n_mat = CDS_obs.shape[1]

    Delta = t_obs[1] - t_obs[0] # Only apprx for now. Move to loop maybe.

    # Only A_trans utilzes new params
    _,Sigma,R_k = build_matrices(lhc_p,sigma,sigma_err,n_mat)

    L = int(m)
    log_likelihood = 0

    # Store predictions.
    pred_Xn = np.zeros(L)
    pred_Pn = np.zeros((L,L))
    Xn = np.zeros((n_obs,L))
    Zn = np.zeros((n_obs,n_mat))
    Pn = np.zeros((n_obs,L,L))

    # For SE calculations, maintain a log likelihood vector.
    log_likelihood = np.zeros(n_obs)

    # Initial Predictions of means and cov
    # Only criteria on mu1 - mu1 < k/kappa. Set to theta
    # This is undeer Q
    mu1 = solve_mu1(kappa_p, theta_p, gamma1,lambda_i=None)
    mu = compute_stationary(kappa_p,theta_p,m,gamma1,mu1 = mu1,lambda_i=None)
    # Bound mu jus in case.
    mu = np.clip(mu,0,1)
    pred_Xn = mu #drift_term(mu,lhc_p,Delta) #mu
    # Just set the covariance guess to the innovated one.
    P_state = np.ascontiguousarray(mu * (1 - mu))
    # Initial Cov Prediction.Z0
    Sigma_sqrt = np.ascontiguousarray(Sigma @ np.diag(np.sqrt(P_state)))
    Sigma_prod = Sigma_sqrt @ Sigma_sqrt.T
    # Just an attemt, not to keep.
    P0 = Sigma_prod.copy()
    pred_Pn = P0[1:,1:] # The above is actually the shape of Y,X of legacy reasons.

    # Compute Psi's for mats. Only prot is a Protection leg is time homogene.
    psi_prot_mat = np.zeros((n_mat, m+1))
    psi_prem_mat = np.zeros((n_mat, m+1))
    psi_prem_pre_mat = np.zeros((n_mat, m+1))
 
    for i in range(n_mat):
        # Pass a scalar from the array X
        psi_prot_mat[i,:] = psi_prot(lhc_q,t_obs[0],t0[0],T_M_grid[i,0])
        psi_prem_pre_mat[i,:] = psi_prem_pre(lhc_q,t_obs[0],t0[0],T_M_grid[i,0])

    psi_prot_mat = np.ascontiguousarray(psi_prot_mat)
    # Verify no change...

    # Run algo.
    for n in range(0,n_obs):
        # for i in range(n_mat):
        # # # Pass a scalar from the array X
        #     psi_prem_mat[i,:] = psi_prem(lhc_q,t_obs[n],t0[n],T_M_grid[i,n])
        psi_prem_mat = psi_prem_fast(lhc_q,t_obs[n],t0[n],T_M_grid[:,n].flatten(),psi_prem_pre_mat)
        psi_prem_mat = np.ascontiguousarray(psi_prem_mat)
        # Extended filter optimized.
        vn,S_k, Xn[n,:], Pn[n,:,:],S_k_inv = update_step_cds(pred_Xn,pred_Pn,cds_fun_mats,cds_deriv_mats,R_k,
                                                             psi_prem_mat,psi_prot_mat,CDS_obs[n,:])
        # Extended kalman filter.
        # vn,S_k, Xn[n,:], Pn[n,:,:],S_k_inv = update_step_cds(pred_Xn,pred_Pn,cds_fun,cds_deriv,R_k,
        #                                                     t_obs[n],t0[n],T_M_grid[:,n],lhc_q,CDS_obs[n,:])
        # Unscented Kalman
        # vn,S_k, Xn[n,:], Pn[n,:,:],S_k_inv = ukf_update_step(pred_Xn,pred_Pn,cds_fun,cds_deriv,R_k,
        #                                                     t_obs[n],t0[n],T_M_grid[:,n],lhc_q,CDS_obs[n,:])


        # ---- 1. state constraint penalty ----
        if np.any(Xn[n, :] < 0) or np.any(Xn[n, :] > 1):
            # clip inside support with small epsilon
            Xn[n, :] = np.clip(Xn[n, :], 1e-10, 1 - 1e-10)
            # penalty += 1e3 * np.sum((Xn[n, :] < 0) | (Xn[n, :] > 1))  # soft penalty

        # Compute Zn too - clearly do outside filter.
        # Xn_extended = np.append([1],Xn[n,:])
        # Zn[n,:] =  cds_fun(lhc_q,Xn_extended,t_obs[n],t0[n],T_M_grid[:,n])


        # ---- 2. innovation covariance stability ----
        det_S = np.linalg.det(S_k)

        # ---- 3. safe inverse / logdet ----
        sign, logdet = np.linalg.slogdet(S_k)
        if sign <= 0:
            logdet = np.log(np.abs(det_S) + 1e-12)

        # ---- 4. likelihood contribution ----
        ll_step = -0.5 * (
            S_k.shape[0] * np.log(2 * np.pi)
            + logdet
            + vn.T @ S_k_inv @ vn
        )

        log_likelihood[n] = ll_step #- penalty


        if (n < n_obs - 1): # Not sensible to predict further.
            Delta = t_obs[n+1] - t_obs[n] # Only apprx for now. Move to loop maybe.
            # Qt needs modification in according to being stat edependent.
            P_state = np.ascontiguousarray(Xn[n, :] * (1 - Xn[n, :]))
            # Sigma_prod = (Sigma @ np.diag(np.sqrt(P_state))) @ (Sigma @ np.diag(np.sqrt(P_state))).T
            Sigma_sqrt = np.ascontiguousarray(Sigma @ np.diag(np.sqrt(P_state)))
            Sigma_prod = Sigma_sqrt @ Sigma_sqrt.T
            Q_k = Sigma_prod[1:,1:].copy() * Delta

            # Then update the predictions:
            # Taylor approximate the transition.
            # Pure Euler
            Xn_n =  np.ascontiguousarray(Xn[n,:])
            pred_Xn = drift_term(Xn_n,lhc_p,Delta)
            # Taylor
            P_cov = drift_deriv_term(Xn_n,lhc_p,Delta)
            Pn_n = np.ascontiguousarray(Pn[n,:,:])
            pred_Pn =  P_cov @ Pn_n @ P_cov.T + Q_k



    return log_likelihood, Xn, Zn, Pn

@njit
def kalman_wrapper(params, t_obs,t0,T_M_grid,CDS_obs,X0,m,r, Y_dim, delta, tenor):
    print(params)
    # For numerical stability.
    # params_p = params[2*m+1:]
    # params_q = params[:2*m+1]
    params_p = params[2*m+1:]
    params_q = params[:2*m+1]
    lambda_i = params_p[:m]
    # kappa, theta, gamma1 = params_q[:m],params_q[m:2*m], params_q[-1]
    kappa, theta,gamma1 = params_q[:m],params_q[m:2*m], params_q[-1]
    kappa_p = kappa - lambda_i
    theta_p = (kappa*theta) / kappa_p
    # theta_p[-1] = theta_p[-1] + lambda_i[-1] / kappa_p[-1]
    # theta_p = theta_p + lambda_i / kappa_p

    lhc_p = rebuild_lhc_struct(kappa_p, theta_p, gamma1, r, Y_dim, delta, tenor, lambda_i=None)
    lhc_q = rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor, lambda_i=None)

    loglik,_, _, _ = kalmanfilter_opt(params, t_obs,t0,T_M_grid,CDS_obs,lhc_p,lhc_q,X0)

    # return negative log likelihood.
    return - np.sum(loglik)

# @njit
def nonlinear_constraints( params, m):
    """
    Vector-valued constraint function for NonlinearConstraint.
    Returns array `cons` such that cons >= 0 are feasible.
    If params can't be unpacked properly, returns a negative array (infeasible).
    """
    m = int(m)
    # expected number of constraints: g1, g2 and 2*m for g3,g4 converted -> total 2 + 2*m
    params_p = np.asarray(params, dtype=float)

    # params_p = params[2*m+1:]
    # params_q = params[:2*m+1]
    params_p = params[2*m+1:]
    params_q = params[:2*m+1]
    # Build constraints: all should be >= 0
    cons = []

    lambda_i,sigma,sigma_err = params_p[:m],params_p[m:2*m],params_p[-1]
    # kappa, theta, gamma1 = params_q[:m],params_q[m:2*m], params_q[-1]
    kappa, theta,gamma1 = params_q[:m],params_q[m:2*m],params_q[-1]
    kappa_p = kappa - lambda_i
    theta_p = (kappa*theta) / kappa_p
    # theta_p[-1] = theta_p[-1] + lambda_i[-1] / kappa_p[-1]
    # theta_p = theta_p + lambda_i / kappa_p

    # ensure arrays
    sigma = np.asarray(sigma)

    # Constraints on lambda. 'Mean reversion needs to be positive'.
    # g1 =  lambda_i[-1] + kappa[-1]*theta[-1]
    # cons.append(np.asarray(g1).flatten())
    # g1 =  lambda_i[-1] + kappa[-1]*theta[-1]
    # cons.append(np.asarray(g1).flatten())

    # Upper bound
    g1 = kappa - lambda_i
    cons.append(g1)


    ### Note, the constraints on theta_p and kappa_p still needs to hold...
    # Drift away from zero sould not be necessary!
    g1 = theta[-1] * kappa[-1] - 0.5 * (sigma[-1]**2)
    cons.append(np.asarray(g1).flatten())

    g1 = theta_p[-1] * kappa_p[-1] - 0.5 * (sigma[-1]**2)
    cons.append(np.asarray(g1).flatten())


    # Drift away from 1. not needed for final
    # Constraint for process m with Y_t
    # if m == 1:
    #     g3 = gamma1 - kappa + kappa * theta 
    #     cons.append(-np.asarray(g3).flatten())
    #     g3 = gamma1 - (kappa_p) + (kappa_p * theta_p) 
    #     cons.append(-np.asarray(g3).flatten())
    # else:
    #     g3 = gamma1 - kappa[:-1] + kappa[:-1] * theta[:-1] + 0.5 * (sigma[:-1] ** 2)
    #     cons.append(-g3)
    #     g3 = gamma1 - (kappa_p[:-1]) + (kappa_p[:-1] * theta_p[:-1]) + 0.5 * (sigma[:-1]**2)
    #     cons.append(-g3)

    g3 = gamma1 - kappa + kappa * theta + 0.5 * (sigma ** 2)
    cons.append(-g3)
    g3 = gamma1 - (kappa_p) + (kappa_p * theta_p) + 0.5 * (sigma**2)
    cons.append(-g3)


    ## All the above are direct artifacts from the model. Still too much flexibility. Choose gamma1
    # s.t. smaller than all thetas.
    # Other constraint in init: that is only for bookkeeping in multivar. Accounted for here
    # in the sigma constraints.

    # Make a theta assumption.
    # for i in range(m):
    #     if i == m-1:
    #         g6 = theta[i] - 1e-4
    #     else:
    #         g6 = theta[i] - theta[i+1] + 1e-4
    #     cons.append(np.asarray(g6).flatten())

    # Gamma constraint.
    # kappa_bound = kappa[0] / (1+kappa[0])
    # kappas = np.min(kappa)
    # g6 = np.minimum(kappa_bound,kappas) - gamma1 # Upper cound on gamma1
    # cons.append(np.asarray(g6).flatten())
    # Assuming increasing gamma.
    # for i in range(m):
    #     if i == m-1:
    #         g6 = kappa[i] - 1e-4
    #     else:
    #         g6 = kappa[i] - kappa[i+1] + 1e-4
    #     cons.append(np.asarray(g6).flatten())

    # stationary_lambda = 0.05 / 100
    # X_dim = kappa.shape[0]
    # if X_dim > 1:
    #     mu = compute_stationary(kappa, theta, 
    #                                         X_dim, gamma1=1, mu1=stationary_lambda, lambda_i=None)
    #     upper_lim_theta_1 =np.minimum(1/2 * mu[1]**(-1),1/2) # always second entrance
    # else: 
    # upper_lim_theta_1 = 1 / 2
    # g5 = upper_lim_theta_1 - theta[0]
    # cons.append(np.asarray(g5).flatten())


    cons = np.concatenate(cons, dtype=float)

    return cons


# @njit
def nonlinear_constraints_mpr( params, m):
    m = int(m)
    # expected number of constraints: g1, g2 and 2*m for g3,g4 converted -> total 2 + 2*m
    params_p = np.asarray(params, dtype=float)

    # params_p = params[2*m+1:]
    # params_q = params[:2*m+1]
    params_p = params[2*m:]
    params_q = params[:2*m]
    # Build constraints: all should be >= 0
    cons = []

    lambda_i,sigma,sigma_err = params_p[:m],params_p[m:2*m],params_p[-1]
    # kappa, theta, gamma1 = params_q[:m],params_q[m:2*m], params_q[-1]
    kappa, theta = params_q[:m],params_q[m:2*m]
    gamma1 = calc_gamma1(kappa,theta)
    kappa_p = kappa - lambda_i
    theta_p = (kappa*theta) / kappa_p
    # theta_p[-1] = theta_p[-1] + lambda_i[-1] / kappa_p[-1]
    # theta_p = theta_p + lambda_i / kappa_p

    # ensure arrays
    sigma = np.asarray(sigma)

    # Constraints on lambda. 'Mean reversion needs to be positive'.
    # g1 =  lambda_i[-1] + kappa[-1]*theta[-1]
    # cons.append(np.asarray(g1).flatten())
    # g1 =  lambda_i[-1] + kappa[-1]*theta[-1]
    # cons.append(np.asarray(g1).flatten())

    # Upper bound
    g1 = kappa - lambda_i
    cons.append(g1)


    ### Note, the constraints on theta_p and kappa_p still needs to hold...
    # Drift away from zero...
    g1 = theta[-1] * kappa[-1] # - 0.5 * (sigma[-1]**2)
    cons.append(np.asarray(g1).flatten())

    g1 = theta_p[-1] * kappa_p[-1] # - 0.5 * (sigma[-1]**2)
    cons.append(np.asarray(g1).flatten())


    # Drift away from 1. not needed for final
    # Constraint for process m with Y_t
    # if m == 1:
    #     g3 = gamma1 - kappa + kappa * theta 
    #     cons.append(-np.asarray(g3).flatten())
    #     g3 = gamma1 - (kappa_p) + (kappa_p * theta_p) 
    #     cons.append(-np.asarray(g3).flatten())
    # else:
    #     g3 = gamma1 - kappa[:-1] + kappa[:-1] * theta[:-1] + 0.5 * (sigma[:-1] ** 2)
    #     cons.append(-g3)
    #     g3 = gamma1 - (kappa_p[:-1]) + (kappa_p[:-1] * theta_p[:-1]) + 0.5 * (sigma[:-1]**2)
    #     cons.append(-g3)

    g3 = gamma1 - kappa + kappa * theta #+ 0.5 * (sigma ** 2)
    cons.append(-g3)
    g3 = gamma1 - (kappa_p) + (kappa_p * theta_p) 
    cons.append(-g3)


    ## All the above are direct artifacts from the model. Still too much flexibility. Choose gamma1
    # s.t. smaller than all thetas.
    # Other constraint in init: that is only for bookkeeping in multivar. Accounted for here
    # in the sigma constraints.

    # Make a theta assumption.
    # for i in range(m):
    #     if i == m-1:
    #         g6 = theta[i] - 1e-4
    #     else:
    #         g6 = theta[i] - theta[i+1] + 1e-4
    #     cons.append(np.asarray(g6).flatten())

    # Gamma constraint.
    # kappa_bound = kappa[0] / (1+kappa[0])
    # kappas = np.min(kappa)
    # g6 = np.minimum(kappa_bound,kappas) - gamma1 # Upper cound on gamma1
    # cons.append(np.asarray(g6).flatten())
    # Assuming increasing gamma.
    # for i in range(m):
    #     if i == m-1:
    #         g6 = kappa[i] - 1e-4
    #     else:
    #         g6 = kappa[i] - kappa[i+1] + 1e-4
    #     cons.append(np.asarray(g6).flatten())

    # stationary_lambda = 0.05 / 100
    # X_dim = kappa.shape[0]
    # if X_dim > 1:
    #     mu = compute_stationary(kappa, theta, 
    #                                         X_dim, gamma1=1, mu1=stationary_lambda, lambda_i=None)
    #     upper_lim_theta_1 =np.minimum(1/2 * mu[1]**(-1),1/2) # always second entrance
    # else: 
    upper_lim_theta_1 = 1 / 2
    g5 = upper_lim_theta_1 - theta[0]
    cons.append(np.asarray(g5).flatten())


    cons = np.concatenate(cons, dtype=float)

    return cons




def equality_constraints( params, m):
    """
    Vector-valued constraint function for NonlinearConstraint.
    Returns array `cons` such that cons >= 0 are feasible.
    If params can't be unpacked properly, returns a negative array (infeasible).
    """
    m = int(m)
    # expected number of constraints: g1, g2 and 2*m for g3,g4 converted -> total 2 + 2*m
    params_p = np.asarray(params, dtype=float)

    params_p = params[2*m+1:]
    params_q = params[:2*m+1]

    # Build constraints: all should be >= 0
    cons = []

    lambda_i,sigma,sigma_err = params_p[:m],params_p[m:2*m],params_p[-1]
    kappa, theta, gamma1 = params_q[:m],params_q[m:2*m], params_q[-1]

    kappa_p = kappa - lambda_i
    theta_p = (kappa*theta) / kappa_p
    # theta_p[-1] = theta_p[-1] + lambda_i[-1] / kappa_p[-1]
    # theta_p = theta_p + lambda_i / kappa_p

    # ensure arrays
    sigma = np.asarray(sigma, dtype=float)
    theta_cons = 1/2 - gamma1 / (4*kappa) - theta
    cons.append(np.asarray(theta_cons).flatten())
    cons = np.concatenate(cons, dtype=float)

    return cons


class LHC_single():
    def __init__(self, r, delta, cds_tenor):
        # Set global params
        self.r = r                  # Set short rate
        self.delta = delta          # Set recovery rate
        self.tenor = cds_tenor      # Set Swap tenor/payments structure.
        self.stationary_lambda = 5 / 10000 # assume long term/stationry default intensity is 5bps

    def initialise_LHC(self, Y_dim, X_dim,X0, rng=None):
        if rng is None:
            rng = np.random.default_rng()  # independent each time

        self.Y_dim, self.m = Y_dim, X_dim
        self.X0 = X0
        self.a = np.ones((self.Y_dim,1))                                      # Y dim is 1 for LHC
        # Set inital values. Need to comply with (38)
        # Then draw kappa. lower bounded by asymption now
        # Sort descending.
        # self.kappa = np.sort(self.kappa)[::-1] # Sort descending
        # self.theta = np.ones(X_dim) * 0.4
        # Same interpretation for 1d-case. For multi dim case - Slightly dif
        # For 1-d case, product of kappa theta needs to equal lambda1
        # self.theta = rng.uniform(0.01, 1 , size=(X_dim,))
        # Assumption to ensure proper initilialization.
        # Kappa can be chosen freely elbeit subject to the 5bps constr. Not strictly necessary anolonger
        self.kappa = rng.uniform(0,1, size=(X_dim,))       # Kappa given
        # Draw gamma1 to get the smallest one possible.
        self.gamma1 = rng.uniform(0,np.min(self.kappa),size= (1,))
        # Theta can be chosen freely for all but i=1.
        self.theta = rng.uniform(0, 1- self.gamma1 / self.kappa , size=(X_dim,))
        
        # The new assumption

        # Build b, beta, A, gamma
        self.rebuild_dynamics()                                     # Build b,beta,gamma again.


    def rebuild_dynamics(self):
        # Formulas, cf p. 16. Only Building Q
        self.b = np.zeros(len(self.theta)).reshape((self.m, self.Y_dim))
        self.b[-1,:] = self.theta[-1] * self.kappa[-1]
        self.beta = np.zeros((self.m, self.m))
        for i in range(0,self.m):
            self.beta[i, i] = - (self.kappa[i])
            if i + 1 < self.m:
                self.beta[i, (i+1)] = self.kappa[i] * self.theta[i]
        # Build gamma. unit vec with gamma 1 in first entry.
        self.gamma = - np.array([[self.gamma1[0]] + [0] * (self.m - 1)]).reshape([1,self.m])

        # In LHC model, gamma is a row vector, b is a column vector for this to make senese.
        self.c = np.zeros(shape=(self.Y_dim, self.Y_dim))

        self.A = np.block([[self.c, self.gamma],
                           [self.b, self.beta]])

        self.id_mat = np.identity(n=self.A.shape[0])
        self.A_star = self.A - self.r * self.id_mat
        self.A_star_inv = np.linalg.inv(self.A_star)


    def flatten_params(self):
        '''
        Extracting the parameters for optimization.
        '''
        self.shapes = {
            'kappa': self.kappa.shape,
            'theta': self.theta.shape,
            'gamma1': self.gamma1.shape,
        }
        return np.concatenate([
            self.kappa.flatten(),
            self.theta.flatten(),
            self.gamma1.flatten(),
        ])

    def unflatten_params(self, flat_vec):
        sizes = {k: np.prod(shape) for k, shape in self.shapes.items()}
        idx = 0
        for key in ['kappa', 'theta', 'gamma1']:
            size = sizes[key]
            shape = self.shapes[key]
            setattr(self, key, flat_vec[idx:idx + size].reshape(shape))
            idx += size
        self.rebuild_dynamics()


    def default_intensity(self,X,Y):
        # This is the form  of the LHC model
        return self.gamma1 * X[0,:]/Y

    def psi_Z(self, t, t_M):
        a_zeros = np.block([self.a, np.zeros(shape=self.m)])
        # return (np.exp(-self.r * (t_M - t)) * a_zeros @ expm(self.A * (t_M - t))).ravel()
        # Note with A_* we can simligy above
        return (a_zeros @ expm(self.A_star * (t_M - t))).ravel()


    def psi_D(self, t, t_M):
        mat_exp = expm(self.A_star * (t_M - t))
        c_gamma = np.block([self.c, self.gamma])
        return - (self.a.T @ c_gamma @ self.A_star_inv @ (mat_exp - self.id_mat)).ravel()

    def psi_D_star(self, t, t_M):
        mat_exp = expm(self.A_star * (t_M - t))
        c_gamma = np.block([self.c, self.gamma])
        return -(self.a.T @ c_gamma @ (
            (t_M - t)* self.A_star_inv @ mat_exp +
            self.A_star_inv @ (self.id_mat * t - self.A_star_inv) @ (mat_exp - self.id_mat)
        )).ravel()

    def psi_prot(self, t, t0, t_M):
        return (1 - self.delta) * (self.psi_D(t, t_M) - self.psi_D(t, t0))

    def psi_prem(self, t, t0, t_M):
        sum_Z = np.zeros(self.Y_dim + self.m)
        sum_D = np.zeros(self.Y_dim + self.m)
        t_grid_len = int(np.floor((t_M - t0) / self.tenor).item()) + 1
        t_grid = np.zeros(t_grid_len)
        for i in range(t_grid_len):
            t_grid[i] = t0 + i * self.tenor
        for j in range(1, t_grid_len):
            dt = t_grid[j] - t_grid[j-1]
            sum_Z += dt * self.psi_Z(t, t_grid[j])
            if j < t_grid_len - 1:
                sum_D += dt * self.psi_D(t, t_grid[j])
        coupon_leg = sum_Z
        accrual_default = (self.psi_D_star(t, t_M) - self.psi_D_star(t, t0))
        accrual_prev = t_grid[-2] * self.psi_D( t, t_M) - sum_D - t0 * self.psi_D( t, t0)

        return (coupon_leg + accrual_default - accrual_prev).flatten()

    def psi_cds(self, t, t0, t_M, k):
        return self.psi_prot(t, t0, t_M) - k * self.psi_prem(t, t0, t_M)


    def CDS_model(self,t_obs, T_M_grid, CDS_obs, t0=None, X_in=None,Y_in=None,Z_in=None,
                  calc_efficiently=False):
        # Get latent states.
        # If t0 is none, assume initial date is today
        if t0 is None:
            t0 = t_obs
        # mat_actual = np.array([[0.2137 + i, 0.4658 + i,0.7178 + i,0.9671+i ]
        #                         for i in range(0,int(np.max(t0)+1))]).flatten()
        # # Ensure mat_actual is sorted
        # mat_actual_sorted = np.sort(mat_actual)

        # # For each element in t_mat_grid, find the smallest mat_actual that is >= element
        # t0 = np.array([mat_actual_sorted[np.searchsorted(mat_actual_sorted, val, side='left')]
        #                                 for val in t0.flatten()]).reshape(t0.shape)

        # Actual maturity dates. Say at March 20, Jun

        kappa, theta, gamma1 = self.kappa,self.theta,self.gamma1[0]
        r = self.r
        Y_dim = self.Y_dim
        delta = self.delta
        tenor = self.tenor
        lhc = rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor)

        # New get states functionality:
        # Numba code to generate matrices to solve for.
        # If nones, need to overwrite
        if (X_in is None) | (Y_in is None) :
            # This function is for the filipovic optiomizatinop
            X_in,Y_in, Z = get_states(lhc, t_obs, T_M_grid, CDS_obs,kappa, theta, gamma1)

        if Z_in is not None:
            X_in,Y_in = self.kalman_X_Y(t_obs,Z_in)

        X,Y = X_in,Y_in

        #print('Done Getting Z,Y,X')
        state_vec = np.vstack([Y, X])

        # Here formula is the t_obs according to formula
        CDS = get_CDS_Model(t_obs, t_obs, T_M_grid, state_vec, lhc,calc_efficiently )
        #print('Done Getting CDS Rate')
        return CDS.T  # NOTE: CHANGED SIGN HERE. No idea why necessary.

    def get_states(self,t_obs, T_M_grid, CDS_obs):
        # Get latent states.
        t0 = t_obs
        # mat_actual = np.array([[0.2137 + i, 0.4658 + i,0.7178 + i,0.9671+i ]
        #                         for i in range(0,int(np.max(t0)+1))]).flatten()
        # # Ensure mat_actual is sorted
        # mat_actual_sorted = np.sort(mat_actual)

        # # For each element in t_mat_grid, find the smallest mat_actual that is >= element
        # t0 = np.array([mat_actual_sorted[np.searchsorted(mat_actual_sorted, val, side='left')]
        #                                 for val in t0.flatten()]).reshape(t0.shape)

        kappa, theta, gamma1 = self.kappa,self.theta,self.gamma1[0]
        r = self.r
        Y_dim = self.Y_dim
        delta = self.delta
        tenor = self.tenor
        lhc = rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor)

        # New get states functionality:
        # Numba code to generate matrices to solve for.
        X,Y,Z = get_states(lhc, t_obs, T_M_grid, CDS_obs,kappa, theta, gamma1)

        return X,Y,Z



    def test_constriants(self):
        # Boolean mask of which constraints are satisfied
        satisfied = self.theta <= 1 - self.gamma1 / self.kappa

        # Indices that FAIL
        failed_idx = np.where(~satisfied)[0]   # ~ flips the booleans

        if failed_idx.size > 0:
            print(f"Constraint failed at indices: {failed_idx}")
            return False
        else:
             #print("All constraints satisfied.")
            return True



    ########### THIS METHODOLODY REBUILDS THE ONE IN ACKERER/FILIPOVIC. #################3
    def objective(self, params, t_obs, T_M_grid, CDS_obs,calc_efficiently=False):
        # Test for feasibility.
        print(params)
        # --------- HARD CONSTRAINT CHECKS ---------
        # 1. Positivity
        # if np.any(params <= 0):
        #     return 1e12  # infeasible, huge loss

        # # 2. Custom constraints
        # g1 = params[self.m-1] * params[2*self.m-1]
        # if g1 < 0:
        #     return 1e12

        # for i in range(self.m):
        #     g3 = params[2*self.m] - params[i] + params[i] * params[self.m+i]
        #     if g3 > 0:
        #         return 1e12


        # Format params for calculations
        # self.test_constriants()
        kappa,theta,gamma1 = params[:self.m], params[self.m:2*self.m] ,params[-1]
        self.unflatten_params(params)


        #  Build Psi functions to avoid redoing it later.
        model_cds = self.CDS_model(t_obs, T_M_grid, CDS_obs,calc_efficiently)
        obj = np.sqrt(0.5 * np.mean((model_cds - CDS_obs)**2))

        return obj

    def build_constraints(self,params):
        # Test for feasibility.
        cons = []
        # 2. Custom constraints
        kappa,theta,gamma1 = params[:self.m],params[self.m:2*self.m], params[-1]
        # set gamma1.
        g1 = kappa[-1]*theta[-1]
        cons.append(np.asarray(g1).flatten())

        for i in range(self.m):
            g3 = -(gamma1 - kappa[i] + kappa[i] * theta[i])
            cons.append(np.asarray(g3).flatten())

        # Add  constraint on theta 1
        # stationary_lambda = 0.05 / 100
        # X_dim = kappa.shape[0]
        # if X_dim > 1:
        #     mu = compute_stationary(kappa, theta, 
        #                                         X_dim, gamma1=1, mu1=stationary_lambda, lambda_i=None)
        #     upper_lim_theta_1 = np.minimum(1/(2 * mu[1]),1/2) # always second entrance
        # else: 
        # upper_lim_theta_1 = 1 / 2
        # g5 = upper_lim_theta_1 - theta[0]
        # cons.append(np.asarray(g5).flatten())

        return  np.array(cons).flatten()



    def optimize_params(self,t_obs, T_M_grid, CDS_obs,base_seed=1000):
        # Retrieve initial parameters.
        flat_init = self.flatten_params().copy()
        # drop gamma1
        CDS_obs = np.ascontiguousarray(CDS_obs)
        # Set optimizer flag to calculate cds spread  efficiently.
        calc_efficient = True
        # result = minimize(
        #     fun = self.objective,
        #     x0 = flat_init,
        #     method='Nelder-Mead',
        #     # method = 'L-BFGS-B', # Finite difference method.
        #     args = (t_obs, T_M_grid, CDS_obs),
        #     options = {
        #         "xatol": 1e-4,
        #         "fatol": 1e-4,
        #         "maxiter": 500,
        #         "disp": True
        #     }
        # )
        bounds = (
                [(1e-6, 2)] * self.m +     # kappa
                [(1e-6,1)] +                # Gamma1
                [(1e-6, 1)] * (self.m)      # theta
                # [(1.e-6,1)]         # gamma1
                )
        constraints = NonlinearConstraint(lambda x: self.build_constraints(x), 0, np.inf)

        # Dont add too much here. Quite quick at finding desired.
        result = differential_evolution(
            func= self.objective,
            # x0=flat_init, # Maybe initial value is wrong here?
            bounds=bounds,
            constraints=(constraints,),
            # args = (t_obs[::5],  T_M_grid[:,::5], CDS_obs[::5,:]),
            args = (t_obs, T_M_grid, CDS_obs,calc_efficient),
            strategy='best1bin',
            popsize=10, #6, # 20 in prod,
            mutation=(0.8, 1.0),
            recombination=0.8,
            maxiter=200, #100, # 500 in prod
            tol=1e-3,
            # workers=1,
            # updating='immediate',
            workers=-1,
            updating='deferred',
            polish=False,
            seed= np.random.RandomState(base_seed) #  Use rng for reproducability.
        )


        # polish_result = minimize(
        #     fun= self.objective,
        #     x0=result.x,  
        #     args = (t_obs, T_M_grid, CDS_obs,calc_efficient),
        #     method='trust-constr',
        #     bounds = bounds,
        #     constraints=(constraints),
        #     options={
        #         'disp': True,
        #         'maxiter': 300  # Give it a reasonable number of iterations
        #     }
        # )

        if result.success:
            print(f"Optimization succeeded, params:{result.x}, objective: {result.fun}")
            self.objective_result = result.fun
        else:
            print("Optimization failed:", result.message)
            self.objective_result = result.fun
        # set gamma1.
        self.unflatten_params(result.x)
        return result.x

    def optimal_parameter_set(self, t_obs,T_M_grid, CDS_obs, base_seed = 1000,  n_restarts = 20):
        # Define grid of values.
        current_objective = 1e10 #very high objective.
        out_params = self.flatten_params()
        for i in range(n_restarts):
            print(f"Optimization {i+1}")
            rng = np.random.default_rng(base_seed + i)  # deterministic but different
            self.initialise_LHC( self.Y_dim, self.m, self.X0, rng=rng)
            self.optimize_params(t_obs, T_M_grid, CDS_obs)
            # Test new constraints
            constrains = self.test_constriants()

            if (self.objective_result < current_objective) & (constrains == True):
                print(f"New optimal parameters at iteration {i+1}.")
                current_objective = self.objective_result
                out_params = self.flatten_params()

        # Set new optimal parameters.
        self.unflatten_params(out_params)

        return out_params



######################### KALMAN FILTER SECTION #########################
    def build_P_params(self,params=None, gamma1=None,rng=None):
        if params is None:
            if rng is None:
                rng = np.random.default_rng()  # independent each time
            Y_dim, X_dim = self.Y_dim, self.m
            gamma1 = self.gamma1[0]

            # Set inital values. Need to comply with (38). enhanced from thm if used in constr
            lambda_i = np.zeros(X_dim)
            for i in range(self.m):
                # if i == self.m - 1:
                #     lambda_i[i] =  rng.uniform(-self.kappa[i]*self.theta[i],
                #                         0.5*(self.kappa[i] - self.kappa[i]*self.theta[i] -
                #                             self.gamma1),size=(1,))
                # else:
                #     lambda_i[i] = rng.uniform(-1,
                #                             (self.kappa[i] - self.theta[i]*self.kappa[i] -
                #                             self.gamma1),size=(1,)) # initialise gamma to comply.
                lambda_i[i] =  rng.uniform(-0.6,
                                        (self.kappa[i] - self.kappa[i]*self.theta[i] -
                                            self.gamma1),size=(1,))
            # Try lambda to zero.
            kappa_p =  self.kappa - lambda_i  #rng.uniform(gamma1, 0.99, size=(X_dim,))       # Kappa given
            theta_p = (self.kappa * self.theta) / kappa_p # Value for prod to be unchanged.
            # # Correction for i=ms
            # theta_p[-1] = theta_p[-1] + lambda_i[-1] / kappa_p[-1]
            # theta_p = theta_p + lambda_i / kappa_p

            ### New stuff: All the ones needed here e.g. sigma, sigma_Err
            sigma_upper_Q = np.sqrt(2*(self.kappa - self.gamma1 - self.kappa*self.theta))
            sigma_upper_P = np.sqrt(2*(kappa_p - self.gamma1 - kappa_p*theta_p))
            sigma_upper = np.minimum(sigma_upper_Q,sigma_upper_P)
            # Final cond in multi setting. Not holding see B1.
            # Actually feller like cond
            sigma_thetakap = np.minimum(np.sqrt(2*self.theta[-1]*self.kappa[-1]),np.sqrt(2*theta_p[-1]*kappa_p[-1]))
            # sigma_thetakap = np.sqrt(2*self.theta[-1]*self.kappa[-1])
            sigma_i = np.zeros(self.m)
            for i in range(0,self.m):
                # not too sure of lower bounds here...
                if i == self.m-1:
                    sigma_i[i] = rng.uniform(0, np.minimum(sigma_thetakap,sigma_upper[i]) , size=(1,))       # Kappa given
                    # sigma_i[i] = rng.uniform(0, sigma_thetakap, size=(1,))       # Kappa given
                else:
                    sigma_i[i] = rng.uniform(0,sigma_upper[i],size=(1,)) # initialise gamma to comply.

            # Sigma error is likely smalll.
            sigma_err = rng.uniform(0.00001, 0.0001, size=(Y_dim,))       # Kappa given

            r = self.r
            Y_dim = self.Y_dim
            delta = self.delta
            tenor = self.tenor
            # Rebuild P parameters.
            lhc = rebuild_lhc_struct(kappa_p, theta_p, gamma1, r, Y_dim, delta, tenor, lambda_i=None)

        else:
            gamma1 = gamma1[0] # due to parametrization
            lambda_i = params[:self.m]
            sigma_i,sigma_err = params[self.m:2*self.m], np.array([params[-1]])

            kappa_p = self.kappa - lambda_i
            theta_p = (self.kappa*self.theta) / kappa_p
            # theta_p[-1] = theta_p[-1] + lambda_i[-1] / kappa_p[-1]
            # theta_p = theta_p + lambda_i / kappa_p

            r = self.r
            Y_dim = self.Y_dim
            delta = self.delta
            tenor = self.tenor
            # Rebuild P parameters.
            lhc = rebuild_lhc_struct(kappa_p, theta_p, gamma1, r, Y_dim, delta, tenor, lambda_i=None)

        self.lambda_i,self.kappa_p,self.theta_p = lambda_i,kappa_p,theta_p
        self.sigma, self.sigma_err= sigma_i, sigma_err

        return lhc

    # Get notmalized drifft stuff.
    def solve_mu1(self,kappa, theta, gamma1, lambda_i=None):

        m = len(kappa)
        if lambda_i is None:
            lambda_i = np.zeros(m)
        def f(mu1):
            prod = 1.0
            for j in range(m):
                prod *= kappa[j] * theta[j] / (mu1 * gamma1 - (kappa[j]-lambda_i[j]))
            return ((-1)**m) * prod - mu1

        # Solve only for admissible range
        return brentq(f, 1e-6, 1.0-1e-6)

    def compute_stationary(self,kappa, theta, m, gamma1, mu1, lambda_i=None):
        mu_process = np.zeros(m)
        if lambda_i is None:
            lambda_i = np.zeros(m)
        for i in range(m-1, -1, -1):
            sign = (-1)**(m - (i+1) + 1)
            prod = 1.0
            for j in range(i, m):
                prod *= kappa[j] * theta[j] / (mu1 * gamma1 - (kappa[j]-lambda_i[j]))
            mu_process[i] = sign * prod
        return mu_process


    ## New: assume long term credit spread 5 bps. Then find gamma1.
    def calc_gamma1(self,kappa, theta, lambda_i=None):
        # Get stationary lambda1. 
        # stationary_spread = 5 / 10000
        # lambda_bar1 = stationary_spread #/ (1-self.delta)
        # m = len(kappa)
        # if lambda_i is None:
        #     lambda_i = np.zeros(m)
        # prod = 1.0
        # for j in range(m):
        #     prod *= kappa[j] * theta[j] / (lambda_bar1 - (kappa[j]-lambda_i[j]))
        # gamma1 = lambda_bar1 / (((-1)**m) * prod)

        # # Solve only for admissible range
        gamma1 = kappa[0] / 2
        # set upper bound to highest implied default intensity times to (MONTE*2)
        # gamma1 = 2*0.16947616891318898
        return np.array([gamma1])


    # One kalman filter for optimizing and one for outputting.

    def get_kalman_params(self, t_obs, T_M_grid, CDS_obs, x0,base_seed,MPR):
        m = self.m
        t0 = t_obs
        if MPR:
            if self.m == 1:
                bounds = (
                    [(1e-6, 2)] * m +     # kappa
                    [(1e-6,1)] * m +     # eta=theta*gamma1. Same bounds. Restriction justified here
                    [(0.0001,1)] +        # gamma1
                    [(-0.6, 0.6)] * m  +         # lambda_p restrict it
                    [(1e-6, 1.5)] * m +     # sigma
                    [(1e-6, 0.01)]         # sigma_err. Unrealistic measure noise is more than 50 Bps
                )
            else:
                bounds = (
                    [(1e-6, 2)] * m +     # kappa. Set to 1
                    [(1e-6, 1)]  +     # eta=theta*gamma1. Same bounds.
                    [(1e-6, 1)] * (m-1) +     # eta=theta*gamma1. Same bounds.
                    [(0.0001,1)] +        # gamma1
                    [(-0.6, 0.6)] * m  +         # lambda_p restrict it
                    [(1e-6, 1.5)] * m +     # sigma
                    [(1e-5, 0.01)]         # sigma_err - very small
                )   
            nlc = NonlinearConstraint(lambda x: nonlinear_constraints(x,self.m), 0, np.inf)

        elif MPR == False:
            if self.m == 1:
                bounds = (
                    [(1e-6, 2)] * m +     # kappa
                    [(1e-6,1)] * m +     # eta=theta*gamma1. Same bounds. Restriction justified here
                    [(0.0001,1)] +        # gamma1
                    [(-0, 0)] * m  +         # lambda_p restrict it
                    [(1e-6, 1.5)] * m +     # sigma
                    [(1e-6, 0.01)]         # sigma_err. Unrealistic measure noise is more than 50 Bps
                )
            else:
                bounds = (
                    [(1e-6, 2)] * m +     # kappa. Set to 1
                    [(1e-6, 1)]  +     # eta=theta*gamma1. Same bounds.
                    [(1e-6, 1)] * (m-1) +     # eta=theta*gamma1. Same bounds.
                    [(0.0001,1)] +        # gamma1
                    [(-0, 0)] * m  +         # lambda_p restrict it
                    [(1e-6, 1.5)] * m +     # sigma
                    [(1e-5, 0.01)]         # sigma_err - very small
                )  
            nlc = NonlinearConstraint(lambda x: nonlinear_constraints_mpr(x,self.m), 0, np.inf)
     
        # theta_index = m   # Index of theta[0]
        # gamma1_index = 2*m  # Index of gamma1[0]

        # # 3. Create the constraint A*x = 0
        # #  This is done as:  1*x[theta_index] - 1*x[gamma1_index] = 0
        # A_matrix = np.zeros(len(bounds))
        # A_matrix[theta_index] = 1.0 / 4
        # A_matrix[gamma1_index] = -1.0
        # equality_con = LinearConstraint(A_matrix, lb=0, ub=0)
        # Give some space to wiggle
        # eq = NonlinearConstraint(lambda x: equality_constraints(x,self.m), -1e-2,1e-2)

        # Global optimizer: few iters. Do on supsed only.
        result = differential_evolution(
            func= kalman_wrapper,
            # x0=x0, # Maybe initial value is wrong here?
            bounds=bounds,
            # constraints=(nlc,eq),
            constraints=(nlc),

            args=(t_obs, t0, T_M_grid, CDS_obs,
                self.X0,self.m,self.r, self.Y_dim, self.delta, self.tenor),
            # args=(t_obs[::5],t0[::5], T_M_grid[:,::5], CDS_obs[::5,:],
            #     self.X0,self.m,self.r, self.Y_dim, self.delta, self.tenor),
            # Giving much more to global optimizer due to more difficult
            strategy='best1bin',        
            popsize=5 * self.m,     
            maxiter=2000,               
            mutation=(0.3, 0.8),        
            recombination=0.7,          
            tol=1e-4,   
            # workers=1,
            # updating='immediate',
            workers=-1,
            updating='deferred',
            polish=False,
            seed= np.random.RandomState(base_seed) #  Use rng for reproducability.
        )

        # If DE failed or hit constraint penalties, bail early
        if result.fun >= 1e12:
            return result.x, 0, 0, 0,0


        polish_result = minimize(
            fun=kalman_wrapper,
            x0=result.x,  # <-- Use DE's best solution as the start
            args=(t_obs, t0, T_M_grid, CDS_obs,
                  self.X0,self.m,self.r, self.Y_dim, self.delta, self.tenor),
            method='trust-constr',
            bounds = bounds,
            constraints=(nlc), # <-- Pass the same nonlinear constraints
            options={
                'disp': False,
                'maxiter': 75,
                'gtol': 1e-4,   
                'xtol': 1e-6,   
                'barrier_tol': 1e-3, 
            }
        )
        # --- Use the best result from the two ---
        # polish_result = result
        if polish_result.success and polish_result.fun < result.fun:
            print(f"--- Local Refinement Succeeded. Final LogLik: {-polish_result.fun} ---")
            optim_params = polish_result.x
            self.kalman_obj = polish_result.fun
        else:
            print(f"--- Local Refinement Failed or did not improve. Using DE result. LogLik: {-result.fun} ---")
            optim_params = result.x
            self.kalman_obj = result.fun


        # --- Evaluate Kalman filter at final parameters ---
        # if self.kalman_obj >= 1e12:
        #     return optim_params, 0, 0, 0,0
        params_p =  optim_params[2*m+1:]
        params_q = optim_params[:2*m+1]
        lambda_i = params_p[:m]
        kappa, theta,gamma1 = params_q[:m],params_q[m:2*m],params_q[-1]

        kappa_p  = kappa - lambda_i
        theta_p = (kappa*theta) / kappa_p
        # theta_p[-1] = theta_p[-1] + lambda_i[-1] / kappa_p[-1]
        # theta_p = theta_p + lambda_i / kappa_p

        lhc_p = rebuild_lhc_struct(kappa_p, theta_p, gamma1, self.r,
                                    self.Y_dim, self.delta, self.tenor,lambda_i=None)
        lhc_q = rebuild_lhc_struct(kappa, theta,  gamma1, self.r,
                                    self.Y_dim, self.delta, self.tenor,lambda_i=None)

        neg_log_lik, Xn, Zn, Pn = kalmanfilter_opt(optim_params, t_obs,t0,T_M_grid,CDS_obs,
                                                    lhc_p,lhc_q,self.X0)
        # Build structures in very end to take onwards.
        self.build_P_params(params_p,np.array([gamma1]))
        self.flatten_params()
        # Add gamma1 to params.
        self.unflatten_params(params_q)
        # Get SE
        kalman_args = (t_obs,t0,T_M_grid,CDS_obs,lhc_p,lhc_q,self.X0)
        se = self.kalman_SE(params_opt=optim_params,f_args=kalman_args,eps=1e-6)
        # Get induced likelihood.
        ll = np.sum(neg_log_lik)
        
        return optim_params, Xn, Zn, Pn, se, ll

    def run_n_kalmans(self, t_obs,T_M_grid, CDS_obs, base_seed = 1000,  n_restarts = 20, MPR=True):
        # Define grid of values.
        current_objective = 1e10 #very high objective.
        out_params = self.flatten_params()
        for i in range(n_restarts):
            print(f"Optimization {i+1}")
            rng = np.random.default_rng(base_seed + i)  # deterministic but different
            # Set Q parameters.
            self.initialise_LHC(self.Y_dim,self.m,self.X0,rng)
            # Get P Parameters /initialise
            lhc_p = self.build_P_params(params=None, gamma1=None,rng=rng)

            # Set the error to be the Stddeviation of CDS_obs
            # Flatten for scipy.
            x0_Q = self.flatten_params()
            # x0_Q = x0_Q[:-1] # No longer gamma interesting.
            x0_P = np.concatenate([
                self.lambda_i.flatten(),
                self.sigma.flatten(),
                self.sigma_err
            ])

            x0 = np.concatenate([x0_Q,x0_P])
            params_p = x0[2*self.m+1:]
            params_q = x0[:2*self.m+1]
            # This still takes on the full vector of params
            lhc_p,lambda_i, sigma, sigma_err = build_P_params(params_p,params_q,lhc_p)
            # kappa, theta, gamma1 = params_q[:self.m],params_q[self.m:2*self.m], params_q[-1]
            kappa, theta = params_q[:self.m],params_q[self.m:2*self.m]

            # Test several random points.
            optim_params,  Xn,Zn, Pn,se,ll= self.get_kalman_params(t_obs,T_M_grid, CDS_obs,x0, base_seed=base_seed + i,MPR=MPR)
        # Set new optimal parameters.
        return  optim_params,  Xn,Zn, Pn,se,ll


    ### Standar error calculation. Run outside as comp needed for each
    def kalman_SE(self,params_opt,f_args, eps=1e-6):
        (t_obs,t0,T_M_grid,CDS_obs,lhc_p,lhc_q,X0) =  f_args
        ll_0,_ ,_,_ = kalmanfilter_opt(params_opt,t_obs,t0,T_M_grid,CDS_obs,lhc_p,lhc_q,X0)
        n = len(ll_0)
        se = np.zeros((params_opt.shape[0],params_opt.shape[0]))
        g = np.zeros((n, params_opt.shape[0]))
        for j in range(len(params_opt)):
            e = np.zeros_like(params_opt)
            e[j] = eps
            # Run filter. Yields Shifted ll for each obs date.
            right_end,_ ,_,_ =  kalmanfilter_opt(params_opt+e, t_obs,t0,T_M_grid,CDS_obs,lhc_p,lhc_q,X0)
            left_end,_ ,_,_=  kalmanfilter_opt(params_opt-e, t_obs,t0,T_M_grid,CDS_obs,lhc_p,lhc_q,X0)
            g[:,j] = (right_end- left_end) / (2*eps)
            # Then compute standard error exact for kalman filter

        # Then loop over dates to compute SE
        for date in  range(n):
            se += np.outer(g[date,:], g[date,:])
        # SE matrix is inverted cov estimate. Asymptotics
        se = np.linalg.pinv(se) 
        se_vec = np.sqrt(np.diag(se))
        return se_vec

    # Transform Kalman Z parameters:
    def kalman_X_Y(self,t_obs,Z):
        n_obs = t_obs.shape[0]
        # find Z0 (use stationary given params)
        #mu1 = self.solve_mu1(self.kappa_p,self.theta_p,self.gamma1)
        #Z0 = self.compute_stationary(self.kappa_p,self.theta_p,self.m,self.gamma1,mu1)
        Y = np.ones(n_obs)
        X = Z # X is a copy of Z
        for i in range(1,n_obs):
            delta = t_obs[i] - t_obs[i-1]
            Y[i] = Y[i-1] + self.gamma @ X[i-1,:] * delta
            X[i] = Y[i] * Z[i,:]
        return X,Y

####### As a consequence of Kalman filtering, we may calculate MPR ###########
    def get_MPR(self, opt_params, Y,X,CDS):
        ## Following appendix B for this.
        Lambda = np.zeros((CDS.shape[0],self.m))
        girsanov = np.zeros((CDS.shape[0],self.m))

        # First, need to build parameters again. Both Q and P params.
        # build parameters if not done yet.
        # Rebuild Q paramters.
        self.rebuild_dynamics()
        # lhc_p
        lhc_p = self.build_P_params(opt_params,np.array([opt_params[2*self.m]]))
        # Get P paramters.
        for n in range(CDS.shape[0]):
            nom = (lhc_p.b - self.b).flatten() * Y[n] + ((lhc_p.beta - self.beta) @ X[:,n])
            denom = self.sigma * np.sqrt(X[:,n] * (Y[n]-X[:,n]))

            # Compute MPR for m
            for m in range(self.m):
                Lambda[n,m] = nom[m] / denom[m]
                girsanov[n,m] = nom[m]

        return Lambda, girsanov


    ###### Montecarlo simulation of processes #############3
    # Below function to simulate the dicretized processes.
    def simul_latent_states(self, chi0, T,M,n_mat,seed=None,scheme='Euler',measure='Q'):
        delta = T / M
        T_return = np.array([0] + [delta*k for k in range(1,M+1)])
        path_Q = np.ones((M + 1, self.m+self.Y_dim))

        # Set initial value.
        path_Q[0,:] = chi0
        W_Q = norm.rvs(size = (M,self.m),random_state=seed) # simulate at beginning - faster!

        # Get A matrix. Add argument if timul under P or Q.
        params_Q = np.concatenate([self.kappa,self.theta,self.gamma1])
        # If params have been sat with meaningfull values
        params_P = np.concatenate([self.lambda_i, self.sigma, self.sigma_err])
        # If explicitly given, manual sigma is used.
        params = np.concatenate([params_Q,params_P])
        # Set also P params.
        lhc_p = self.build_P_params(params_P,self.gamma1)
        A_p,cov_trans,_ = build_matrices(lhc_p,self.sigma,self.sigma_err,n_mat)

        self.rebuild_dynamics()
        # Get A
        if measure == 'Q':
            A = self.A
        elif measure == 'P':
            A = A_p
        if scheme == 'Euler':
            for i in range(1,M+1):
                # Cap X process in 0,1 to avioid discretization error.
                path_Q[i-1,1:] = np.clip(path_Q[i-1,1:],1e-8,path_Q[i-1,0]-1e-8)
                # Out
                mu_t = A @ path_Q[i - 1,:]
                # Create Sigma:
                P_state = np.array(path_Q[i - 1,1:] * (path_Q[i-1,0] - path_Q[i-1,1:]))
                Sigma_prod = (cov_trans @ np.diag(np.sqrt(P_state)))
                path_Q[i,:] = path_Q[i-1,:] + delta*mu_t +  np.sqrt(delta) * Sigma_prod @ W_Q[i-1,:]

        if scheme == 'Milstein':
            for i in range(1,M+1):
                path_Q[i-1,1:] = np.clip(path_Q[i-1,1:], 1e-8,path_Q[i-1,0]-1e-8)

                mu_t = A @ path_Q[i - 1,:]
                # Create Sigma:
                P_state = np.array(path_Q[i - 1,1:] * (path_Q[i-1,0] - path_Q[i-1,1:]))
                Sigma_prod = (cov_trans   @ np.diag(np.sqrt(P_state)))
                P_state_mil =  np.diag(np.array(1/(2*np.sqrt(path_Q[i - 1,1:] * (path_Q[i-1,0] - path_Q[i-1,1:])))
                                       * (path_Q[i-1,0] - 2*path_Q[i-1,1:])))

                sigma_prime_t = cov_trans[1:,: ]  @ P_state_mil

                # sigma to mult on BM
                sigma_mil_with0 = np.zeros((self.m+1,self.m+1))
                sigma_mil_with0[1:,1:] = sigma_prime_t

                # Out
                path_Q[i,:] = (path_Q[i-1,:] + delta*mu_t +
                               np.sqrt(delta) * Sigma_prod @ W_Q[i-1,:] +
                               1/2 * delta* sigma_mil_with0 @ Sigma_prod @ (W_Q[i-1,:]**2-1))

        return T_return, path_Q

    def simul_Z(self, chi0, T,M,n_mat,seed=None,scheme='Euler', measure = 'Q'):
        delta = T / M
        T_return = np.array([0] + [delta*k for k in range(1,M+1)])
        path_Q = np.ones((M + 1, self.m))
        # Set initial value.
        path_Q[0,:] = chi0
        W_Q = norm.rvs(size = (M,self.m),random_state=seed) # simulate at beginning - faster!
        # Get A matrix. Add argument if timul under P or Q.
        params_Q = np.concatenate([self.kappa,self.theta,self.gamma1])
        # If params have been sat with meaningfull values
        params_P = np.concatenate([self.lambda_i, self.sigma, self.sigma_err])
        # If explicitly given, manual sigma is used.
        params = np.concatenate([params_Q,params_P])
        # Set also P params.
        lhc_p = self.build_P_params(params_P,self.gamma1)
        A_p,cov_trans,_ = build_matrices(lhc_p,self.sigma,self.sigma_err,n_mat)

        if measure == 'Q':
            b, beta = self.b,self.beta
        elif measure == 'P':
            b, beta = A_p[self.Y_dim:, :self.Y_dim], A_p[self.Y_dim:,self.Y_dim:]

        self.rebuild_dynamics()
        if scheme == 'Euler':
            for i in range(1,M+1):
                # Out
                path_Q[i-1,:] = np.clip(path_Q[i-1,:], 1e-8,1-1e-8)
                diag_term =  np.identity(self.m)*(-self.gamma.flatten() @path_Q[i - 1,:])
                mu_t = (b.flatten()+
                        (beta + diag_term) @ path_Q[i - 1,:])
                # Create Sigma:
                P_state = np.array(path_Q[i - 1,:] * (1 - path_Q[i-1,:]))
                Sigma_prod = (cov_trans[1:,: ] @ np.diag(np.sqrt(P_state)))
                path_Q[i,:] = path_Q[i-1,:] + delta*mu_t +  np.sqrt(delta) * Sigma_prod @ W_Q[i-1,:]

        if scheme == 'Milstein':
            for i in range(1,M+1):
                path_Q[i-1,:] = np.clip(path_Q[i-1,:],1e-8,1-1e-8)

                diag_term =  np.identity(self.m)*(-self.gamma.flatten() @path_Q[i - 1,:])
                mu_t = (b.flatten()+
                        (beta + diag_term) @ path_Q[i - 1,:])
                # Create Sigma:
                P_state = np.array(path_Q[i - 1,:] * (1 - path_Q[i-1,:]))
                Sigma_prod = (cov_trans[1:,: ] @ np.diag(np.sqrt(P_state)))
                P_state_mil =  np.diag(np.array(1/(2*np.sqrt(path_Q[i - 1,:] * (1 - path_Q[i-1,:])))
                                       * (1 - 2*path_Q[i-1,:])))

                sigma_prime_t = cov_trans[1:,: ]  @ P_state_mil

                # sigma to mult on BM

                # Out
                path_Q[i,:] = (path_Q[i-1,:] + delta*mu_t +
                               np.sqrt(delta) * Sigma_prod @ W_Q[i-1,:] +
                               1/2 * delta* sigma_prime_t @ Sigma_prod @ (W_Q[i-1,:]**2-1))

        return T_return, path_Q



    # Simulate option prices in the model.
    def get_cdso_price_MC(self,t,t0,t_M,strikes,chi0,N,M,seed=1000, P_params=None):
        # If p params specific (Sigma specific), se can calculate for differnt sigma
        if P_params is not None:
            lhc_p = self.build_P_params(P_params,self.gamma1)

        # Get LHC_Q for pricing
        lhc = rebuild_lhc_struct(self.kappa, self.theta, self.gamma1[0], self.r,
                            self.Y_dim, self.delta, self.tenor)

        # N prices are comuted and averaged MC
        N_strikes = strikes.shape[0]
        prices = np.zeros(shape = (N,N_strikes))
        prices_MC_hist = np.zeros(shape = (N,N_strikes))
        for i in range(N):
            # Get Latent states. Simulate to time of inception of CDS. Calculate only for 1 maturity.
            # T_return, X_Q = self.simul_latent_states( chi0,t0,M,n_mat=1,seed=seed,scheme='Milstein')
            T_return, X_Q = self.simul_latent_states( chi0,t0,M,n_mat=1,seed=seed,
                                                     scheme='Milstein', measure='Q')

            S = X_Q[:,0]
            # Determine if default or not at t0. If S_t \leq unif(1) option payoff is zero.
            # U = uniform.rvs(random_state=seed)
            # # If survival falls below U at any points, default happened prior to t0
            # if np.any(S <= U):
            #     prices[i] = 0
            # Else - begin to compute prices as no default
            # else:
            latent_end = X_Q[-1,:]
            # Value is assumed to be exactly at inception date first date of contract
            # Loop over strikes here, to save simul time later on (also same randomness)
            for j in range(N_strikes):
                # Find value of option using the strike. This is the payoff.
                Value_CDS = psi_cds(lhc,t0, t0, t_M, strikes[j]) @ latent_end 
                # Discount back. Also, divide by a^TY_t. If t=0, then not matter.
                # Note still an option, so only enter if positive.
                prices[i,j] = np.exp(-self.r * (t0 - t)) * np.maximum(Value_CDS,0) / S[0]
            # Achieve a running mean also for convergence assessment.
            prices_MC_hist[i, :] = np.mean(prices[:i+1, :], axis=0)
            seed += 1
        print(f'CDSO done')
        price_MC = np.mean(prices,axis=0)

        return prices_MC_hist,price_MC


    # Simulate digital Barrier in the model.
    def get_digital_barrier_price_MC(self,t,t0,t_M,T,barriers,chi0,N,M,seed=1000, P_params=None):
        '''
        t: Time to price at
        t0: Start of CDS.
        t_M: Maturity of CDS
        T: Maturity of option. Needs to satisfy T<t_m
        '''
        # If p params specific (Sigma specific), se can calculate for differnt sigma
        if P_params is not None:
            lhc_p = self.build_P_params(P_params,self.gamma1)

        # N prices are comuted and averaged MC
        N_strikes = barriers.shape[0]
        prices = np.zeros(shape = (N,N_strikes))
        test_MC =  np.zeros(shape = (N,N_strikes))
        prices_MC_hist = np.zeros(shape = (N,N_strikes))
        lhc = rebuild_lhc_struct(self.kappa, self.theta, self.gamma1[0], self.r,
                            self.Y_dim, self.delta, self.tenor)
        for i in range(N):
            # Get Latent states. Simulate path of CDS till mat.
            T_return, X_Q = self.simul_latent_states( chi0,T,M,n_mat=1,seed=seed,
                                                     scheme='Milstein', measure='Q')
            Xn = X_Q[:,self.Y_dim:]
            Yn = X_Q[:,0]
            # Retrieve price path...
            T_M_grid = (np.array([t_M]*T_return.shape[0])).reshape((1,T_return.shape[0]))
            t0_arr = (np.array([t0]*T_return.shape[0])).reshape((T_return.shape[0]))
            # Here formula is the t_obs according to formula
            CDS_sim = get_CDS_Model(T_return, t0_arr, T_M_grid, X_Q.T, lhc )
            S = Yn
            # Determine if default or not at t0. If S_t \leq unif(1) option payoff is zero.
            U = uniform.rvs(random_state = seed)
            # If survival falls below U at any points, default happened prior to T.
            # Should be zero, but depends on barrier. Default happens at some point below..
            default_event =  S <= U
            if np.any(default_event):
                # In this instance, default has happened as some point. Find index.
                idx = np.argmax(default_event==1)
                # Get path maximum up to the point:
                max_cds_to_default = np.max(CDS_sim[:idx+1])
                b_idx = np.where(max_cds_to_default >= barriers)
                prices[i,b_idx] = np.exp(-self.r * (T - t))

            # Else - no default happened, but same logic as before.
            else:
                # Get path maximum up to expiry:
                max_cds_to_default = np.max(CDS_sim)
                b_idx = np.where(max_cds_to_default >= barriers)
                prices[i,b_idx] = np.exp(-self.r * (T - t))
            b_idx = np.where(np.max(CDS_sim) >= barriers)
            prices_MC_hist[i, :] = np.mean(prices[:i+1, :], axis=0)
            seed += 1
        print(f'Digital done')
        price_MC = np.mean(prices,axis = 0)
        return prices_MC_hist,price_MC

    # Simulate Lookback in the model.
    def get_lookback_price_MC(self,t,t0,t_M,T,chi0,N,M,seed=1000, P_params=None):
        '''
        t: Time to price at
        t0: Start of CDS.
        t_M: Maturity of CDS
        T: Maturity of option. Needs to satisfy T<t_m
        '''
        prices = np.zeros(shape = N)
        prices_MC_hist = np.zeros(shape = N)
        cds_min =  np.zeros(shape = N)
        # If p params specific (Sigma specific), se can calculate for differnt sigma
        if P_params is not None:
            lhc_p = self.build_P_params(P_params,self.gamma1)
        lhc = rebuild_lhc_struct(self.kappa, self.theta, self.gamma1[0], self.r,
                            self.Y_dim, self.delta, self.tenor)
        # N prices are comuted and averaged MC
        prices = np.zeros(shape = N)
        for i in range(N):
            # Get Latent states. Simulate path of CDS till mat.
            T_return, X_Q = self.simul_latent_states( chi0,T,M,n_mat=1,seed=seed,
                                                     scheme='Milstein', measure='Q')
            # Retrieve price path...
            T_M_grid = (np.array([t_M]*T_return.shape[0])).reshape((1,T_return.shape[0]))
            t0_arr = (np.array([t0]*T_return.shape[0])).reshape((T_return.shape[0]))
            # Here formula is the t_obs according to formula
            CDS_sim = get_CDS_Model(T_return, t0_arr, T_M_grid, X_Q.T, lhc )
            if np.any(CDS_sim == np.nan):
                print('Nan value')

            S = X_Q[:,0]
            # Determine if default or not at t0. If S_t \leq unif(1) option payoff is zero.
            U = uniform.rvs(random_state = seed)
            # If survival falls below U at any points, default happened prior to T - No payoff
            # No payoff makes sense as payoff contigent on final price.
            default_event =  S <= U
            if np.any(default_event):
                prices[i] = 0
                # set min to zero here
                cds_min[i] = 0  # np.min(CDS_sim[:i])
            # Else - no default happened
            else:
                # Get path minimum of CDS:
                cds_min[i]  = np.min(CDS_sim)
                prices[i] = np.exp(-self.r * (T - t)) * (CDS_sim.flatten()[-1] - cds_min[i] )
            prices_MC_hist[i] = np.mean(prices[:i+1])

            seed += 1
        print(f'Loockback done')
        # Final price
        price_MC = np.mean(prices)

        return prices_MC_hist,price_MC,cds_min


########### OPTION APPROXIMATION FORMULAS on credit default swaps. ###################

    def get_bBounds(self, t0, t_M,k):
        lhc_q = rebuild_lhc_struct(self.kappa, self.theta, self.gamma1[0], self.r,
                                   self.Y_dim, self.delta, self.tenor)
        b_min = np.sum(np.minimum(psi_cds(lhc_q,t0,t0,t_M,k),np.zeros(self.m+1)))
        b_max = np.sum(np.maximum(psi_cds(lhc_q,t0,t0,t_M,k),np.zeros(self.m+1)))

        return b_min,b_max


    def f_n(self,n,t,t0,b_min,b_max,Y):
        Lint, _ = quad(
            lambda x: (x * self.gen_legendre_poly(x,n,b_min,b_max)),
            0, b_max,
            limit=1000,
            epsabs=1e-8,
            epsrel=1e-8
        )
        return np.exp(-self.r * (t0-t)) * Lint/ Y



    ######## Then, we need some algoritm to compute the value of the matrix G

    # Assume it is the inital price i need in Legendre poly.
    def PriceCDS(self,z, n,t,t0,t_M,k,Y):
        b_min,b_max = self.get_bBounds(t0, t_M,k)

        pi = 0
        # Loop from 0 to n+1 (n)
        for j in range(n+1):
            f_j = self.f_n(j,t,t0,b_min,b_max,Y)
            GLPoly = self.gen_legendre_poly(z,j,b_min,b_max)
            pi += f_j * GLPoly

        return pi


#### Get coefficients to write (44) as a polynomium ni Z.
# Get coefficients of P_n in monomial basis
    def get_legendre_coeffs(self,n_max):
        # Determine maximum monomial length
        a_max = n_max + 1  # max degree + 1 for coefficients
        a = np.zeros((n_max + 1, a_max))

        for n in range(n_max + 1):
            P = Legendre.basis(n)
            coef = P.convert(kind=np.polynomial.Polynomial).coef
            a[n, :coef.shape[0]] = coef  # fill only existing coefficients

        return a

    def get_scaled_legendre_coeffs(self, n_max, b_min, b_max):
        mu = 0.5 * (b_max + b_min)
        sigma = 0.5 * (b_max - b_min)
        a_max = n_max + 1
        a_scaled = np.zeros((n_max + 1, a_max))

        # affine map x = (z - mu)/sigma
        p_affine = np.polynomial.Polynomial([-mu / sigma, 1.0 / sigma])

        for n in range(n_max + 1):
            # Legendre P_n(x)
            Pn = Legendre.basis(n).convert(kind=np.polynomial.Polynomial)
            # Compose P_n((z-mu)/sigma)
            Pn_scaled = Pn(p_affine)
            # correct normalization for orthonormal basis on [b_min,b_max]
            # Veryfi the equation of generalized Legendre poly?
            norm = np.sqrt((2 * n + 1) / ((b_max - b_min)))
            # norm = np.sqrt((2 * n + 1) / (2*(sigma**2)))

            coef = norm * Pn_scaled.coef
            a_scaled[n, :coef.shape[0]] = coef

        return a_scaled
    # utilize numpy for legendre.
    def legendre_poly(self,x, n):
        # Coeff vector with 1 at index n
        c = np.zeros(n + 1)
        c[n] = 1.0
        return legval(x, c)


    def gen_legendre_poly(self,x, n, b_min, b_max):
        mu = 0.5 * (b_max + b_min)
        sigma = 0.5 * (b_max - b_min)

        z = (x - mu) / sigma
        # norm = np.sqrt((1 + 2 * n) / (2 * sigma**2))
        norm = np.sqrt((1 + 2 * n) / ( (b_max - b_min)))

        return norm * self.legendre_poly(z, n)

    ### What one would need now, is to compute moments of payoff corresponding
    ## To monomial basis in a matrix.

    ##### Functionality for expectations. Likely numba for efficiency.
        ### Compute conditional moments of Y and X.
    def monomial_basis(self,y,X,poly_deg):
        m = X.shape[0]
        N_n = math.comb(poly_deg+1+m,poly_deg)
        stacked = np.append(y,X)
        basis = np.ones(N_n)
        vars = ['y'] + [f'X{i}' for i in  range(1,m+1) ]

        if poly_deg == 0:
            return basis
        elif poly_deg == 1:
            basis[1:] = stacked
            mono_map = ['1','y'] + [f'X{i}' for i in range(1,m+1) ]
            # Index.


        elif poly_deg == 2:
            # Same as in n=1
            basis[1:(m+1)+1] = stacked
            # then add cross terms starting from y.
            mono = list(combinations_with_replacement(stacked, poly_deg))
            mono_map = ['1','y'] + [f'X{i}' for i in range(1,m+1) ]
            mono_map_iter = list(combinations_with_replacement(vars, poly_deg))
            for i in range(0,len(mono)):
                basis[(m+1)+1+i] = mono[i][0]*mono[i][1]
                mono_map.append(mono_map_iter[i][0]+mono_map_iter[i][1])

        term_index = {term: i for i, term in enumerate(mono_map)}

        return basis,term_index


    # Then, we are ready for logic to compute this G_n thingy.
    def poly_G(self,m,poly_deg,index):
        # get x dim and number of monomial basis
        N_n = math.comb(poly_deg+1+m,poly_deg)
        ### Then ready to compute. Only reasonable for n>0.
        G_n = np.zeros(shape = (N_n,N_n))
        if poly_deg == 1:
            # Get indices to fill:
            # y parts.
            G_n[index['X1']:,index['y'] ] = -self.gamma

            # X col part.
            G_n[index['y']:,index['X1']: ] = np.vstack([self.b.T,self.beta.T])


        elif poly_deg == 2:
            ### First part is the same as earlier -> same rows and entries
            ##### Mean section
            # y col
            G_n[index['X1']:index['yy'],index['y'] ] = self.gamma
            # x col
            G_n[index['y']:index['yy'],index['X1']: index['yy']] = np.vstack([self.b.T,self.beta.T])

            ##### Second order section
            ### When Poly is y**2:
            # index y^2 col is m+1+1.
            G_n[index['yX1']:index['X1X1'],index['yy']] =  2*self.gamma

            ### Polynomia yx_i i.e. Cross terms.
            # beta part...
            G_n[index['yy']:index['X1X1'],index['yX1']:index['X1X1']] =  np.concatenate([self.b.T,self.beta.T])
            # Cross term with gammas (utilize lhc form).
            gamma_flat = self.gamma.flatten()

            for k in range(1, m + 1): # Loop over columns: yX_k
                col_key = f'yX{k}'

                for i in range(1, m + 1): # Loop over rows: X_i X_k
                    # Handle ordering, e.g., 'X1X2' not 'X2X1'
                    row_key = f'X{i}X{k}' if i <= k else f'X{k}X{i}'

                    if row_key in index:
                        # G_n[row, col] = coeff
                        # G_n[X_i X_k, yX_k] = gamma_i
                        G_n[index[row_key], index[col_key]] += gamma_flat[i-1]

            index_to_key = {v: k for k, v in index.items()}
            ### X cross terms - i.e. cols with XiXj for i!=j
            for idx in range(index['X1X1'],N_n):
                i, j = map(int,  re.findall(r'\d+', index_to_key[idx]))
                # Fill out mixed deriv cols. yx_i,x1xi,.., xixm
                if (i<j):
                    G_n[index[f'yX{i}'],index[f'X{i}X{j}']] +=  self.b.flatten()[j-1]
                    G_n[index[f'yX{j}'],index[f'X{i}X{j}']] +=  self.b.flatten()[i-1]

                    # X cross contributions for all k
                    for k in range(1, m+1):
                        key_i = f'X{i}X{k}' if i <= k else f'X{k}X{i}'
                        key_j = f'X{j}X{k}' if j <= k else f'X{k}X{j}'
                        if key_i in index:
                            G_n[index[key_i], index[f'X{i}X{j}']] += self.beta.T[ k-1,j-1]
                        if key_j in index:
                            G_n[index[key_j], index[f'X{i}X{j}']] += self.beta.T[k-1,i-1]

                # The double terms are what we consider now
                if i == j:
                    G_n[index[f'yX{i}'],index[f'X{i}X{i}']] += self.sigma[i-1]**2
                    G_n[index[f'X{i}X{i}'],index[f'X{i}X{i}']] += - self.sigma[i-1]**2

                    # Remaining terms.
                    G_n[index[f'yX{i}'],index[f'X{i}X{i}']] += 2 * self.b.flatten()[i-1]

                    for run_idx in range(1,m+1):
                        # logic to flip
                        if run_idx > i:
                            G_n[index[f'X{i}X{run_idx}'],index[f'X{i}X{i}']] += 2 *  self.beta.T[run_idx-1,i-1]
                        else:
                            G_n[index[f'X{run_idx}X{i}'],index[f'X{i}X{i}']] +=  2 * self.beta.T[run_idx-1,i-1]

        return G_n

    #### Calculate moments/covariance.

    def moments_poly(self,y,X,poly_deg,p_idxs,time_delta,G_n,basis):
        ### Get monomial basis.
        m = X.shape[0]
        N_n = math.comb(poly_deg+1+m,poly_deg)
        p_coord = np.zeros( N_n)
        p_coord[p_idxs] = 1

        return basis @ expm(G_n * time_delta) @ p_coord

    ### this iterrative one is probably best to compute in numba...

    def h_poly(self,alpha,s,x,chi_sym):
        '''
        Computes polynomia h given enumeration.
        '''
        value =  s**alpha[0] * np.prod(x**alpha[1:])

        chi_sym_out =  chi_sym[0]**alpha[0] * np.prod(chi_sym[1:]**alpha[1:])

        return value, chi_sym_out

    def compute_enumerations(self,m,n):
        '''
        Compute the possible enummerations of degree n.
        '''
        alphas = [alpha for alpha in product(range(n+1), repeat=m+1) if sum(alpha) <= n]
        alphas = np.array(alphas).T
        return alphas

    #### Compute expected value and c_pi according to Lemma 4.4

    def get_cdso_price(self,t,t0,t_M,Y_t,X_t,strikes,n_max):
        # Get matrix of a_coeffs
        prices = np.zeros(strikes.shape[0])
        moments = np.ones((strikes.shape[0],n_max+1))
        alphas = self.compute_enumerations(m=self.m, n=n_max)
        chi_syms = sp.symbols(f'Y X1:{self.m+1}')
        chi_sym = sp.Matrix(chi_syms)
        ### Precompute a lot of heavy stuff.
        # get the index vector
        a = self.A @ chi_sym
        sigma_ext = sp.Matrix([0, *self.sigma])   # prepend 0 symbolically
        diag_entries = [sp.sqrt(chi_sym[i] * (chi_sym[0] - chi_sym[i])) * sigma_ext[i] for i in range(self.m+1)]
        b = sp.diag(*diag_entries)
        # As second term in drift, need monomials to fourth degree.
        polyclass = mpg(a, b, chi_sym, max_degree=n_max)
        ### Get moment matrices
        start = time.time()
        G_n = polyclass.generator_matrix()
        end = time.time()
        print(f"Time of G_N {end-start}, shape={G_n.shape[0]}")
        mat_expo = expm(G_n*(t0-t))
        chi0 =  np.append([Y_t],X_t)
        chi_dict = dict(zip(chi_syms,chi0))
        basis = polyclass.basis_mat.subs(chi_dict).evalf()

        # Create touple with alphas and indices
        idx_col_map = np.zeros(alphas.shape[1])
        for cols in range(alphas.shape[1]):
            poly, chi_ennum = self.h_poly(alphas[:,cols],Y_t,X_t,chi_syms)
            idx_col_map[cols]  = polyclass.basis.index(chi_ennum)

        lhc = rebuild_lhc_struct(self.kappa,self.theta,self.gamma1[0],self.r,self.Y_dim,
                                 self.delta,self.tenor,self.lambda_i)
        for i,k in enumerate(strikes):
            # Loop from 0 to n+1 (n)
            b_min,b_max = self.get_bBounds(t0, t_M,k)
            legendre_coeff = self.get_scaled_legendre_coeffs(n_max,b_min,b_max)
            # Using recursion, should be calculated here. 
            c_pi_val = np.zeros(alphas.shape[1])
            moments[i,:] = moments_Z(alphas,t,t0,t_M,k,y=Y_t,X=X_t,n_max=n_max,
                                          basis=np.array(basis,dtype = float),
                                          mat_expo=mat_expo,idx_col_map=idx_col_map,lhc=lhc,c_pi_val=c_pi_val)
            # Finally do the legendre approximatino
            for j in range(n_max+1):
                f_j = self.f_n(j,t,t0,b_min,b_max,Y_t)
                GLPoly = legendre_coeff[j,:] @ moments[i,:]
                prices[i] += f_j * GLPoly
                print(f'Done with strike {k}, n={j}')

            print(f'Done with strike {k}')
        return prices



### compute c_pi in numba for faster comp.
# @njit
def c_pi(alphas,alpha,psi_cds_val,c_pi_val):
    # Base case: alpha all zeros
    if np.sum(alpha) == 0:
        return 1.0  # or 1.0 if needed by context

    # Base case: degree 1
    if np.sum(alpha) == 1:
        idx = np.argmax(alpha)
        return psi_cds_val[idx]

    # Recursive case: sum alpha > 1. Use previous vals
    c = 0.0
    d = len(alpha)
    for i in range(d):
        if alpha[i] -1 >=0 :
            alpha_minus = alpha.copy()
            alpha_minus[i] -= 1
            idx = np.argmax(np.sum(alphas==np.array([alpha_minus]).T,axis=0)==psi_cds_val.shape[0])
            c += psi_cds_val[i] * c_pi_val[idx]
    return c



# Sugegstio for faster imp.
# @njit
def c_pi_fast(alpha, psi):
    # compute the demoninator of the multinomial coefficient
    deg = 0
    denominator = 1
    # loop over the alpha in question
    for i in range(alpha.size):
        # Ingrement to get the degree of alpha
        deg += alpha[i]
        # factorial of alpha[i]
        alpha_i = alpha[i]
        alpha_i_fac = 1
        # Factorial of alpha_i
        for k in range(1, alpha_i+1):
            alpha_i_fac *= k
        # add to toal 
        denominator *= alpha_i_fac

    # factorial of the degree of the multindex
    numerator = 1
    for i in range(1, deg+1):
        numerator *= i

    multinomial_coeff = numerator / denominator

    # Compute psi monomial product 
    prod = 1.0
    for i in range(alpha.size):
        alpha_i = alpha[i]
        prod *= psi[i]**alpha_i

    return multinomial_coeff * prod

# can work for n=1
# @njit
def moments_Z(alphas,t,t0,t_M,k,y,X,n_max,basis,mat_expo,idx_col_map,lhc,c_pi_val):
    ### Loop over alpha indices. These correspond to desired basis.
    ### Compute CDS spread moments.
    # psi_cds_vec = self.psi_cds(t0,t0,t_M,k)
    psi_cds_vec = psi_cds(lhc,t0,t0,t_M,k)

    moments = np.zeros(n_max+1)

    # moment loop wrapped in numba
    for n in range(0,n_max + 1):
        # Find cols of alphas that should be summed.
        idx_sum = np.where(np.sum(alphas,axis=0)==n)[0]
        alphas_comp = alphas[:,idx_sum]
        
        for idx in range(alphas_comp.shape[1]):
            ## Fill out values
            # start = time.time()
            # curr_index = np.argmax(np.sum(
            #     alphas==np.array([alphas_comp[:,idx]]).T,axis=0)==psi_cds_vec.shape[0])
            # c_pi_val[curr_index] = c_pi(alphas,alphas_comp[:,idx],
            #                                                             psi_cds_vec,c_pi_val)
            # poly, chi_ennum = self.h_poly(alphas_comp[:,idx],y,X,chi_syms)
            # c_pi_val = c_pi(alphas_comp[:,idx],psi_cds_vec)
            # end = time.time()
            # print(f'Time of recursion method: {end-start}')
            # start = time.time()
            c_pi_val2 = c_pi_fast(alphas_comp[:,idx],psi_cds_vec)
            # end = time.time()
            # print(f'Time of new method: {end-start}')

            # Get the index of the expectation to compute. Generally works for cond exp
            # e_i = np.zeros(shape = len(basis_class))
            e_i = np.zeros(shape = basis.shape[0])

            # e_i[basis_class.index(chi_ennum)] = 1
            e_i[np.int16(idx_col_map[idx_sum[idx]])] = 1

            term = basis.T @ mat_expo @ e_i

            moments[n] += c_pi_val2 * term[0]

    return moments


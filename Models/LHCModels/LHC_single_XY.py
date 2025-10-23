import numpy as np 
import math
from scipy.optimize import minimize, NonlinearConstraint , Bounds
from scipy.stats import norm, ncx2, gamma, expon, uniform
from scipy.integrate import solve_ivp 
from scipy.linalg import expm 
from itertools import combinations_with_replacement, product
import re
from scipy.optimize import lsq_linear 
from numba import njit, float64, int64 
from numba.experimental import jitclass 
from scipy.integrate import quad
import copy
from scipy.linalg import sqrtm
from scipy.optimize import brentq
from scipy.optimize import differential_evolution
from types import SimpleNamespace


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

@jitclass(spec)
class LHCStruct:
    def __init__(self, a, c, gamma, b, beta, A, A_star, A_star_inv, id_mat, r, m, Y_dim, delta, tenor):
        self.a = np.ascontiguousarray(a)
        self.c = np.ascontiguousarray(c)
        self.gamma = np.ascontiguousarray(gamma)
        self.b = np.ascontiguousarray(b)
        self.beta = np.ascontiguousarray(beta)
        self.A = np.ascontiguousarray(A)
        self.A_star = np.ascontiguousarray(A_star)
        self.A_star_inv = np.ascontiguousarray(A_star_inv)
        self.id_mat = np.ascontiguousarray(id_mat)
        self.r = r
        self.m = m
        self.Y_dim = Y_dim
        self.delta = delta
        self.tenor = tenor

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
def mat_exp_approx(A, dt, tol=1e-10):
    n = A.shape[0]
    I = np.eye(n)
    Adt = A * dt

    mat_expo = I.copy()
    term = I.copy()
    
    # We use a fixed upper limit to prevent infinite loops in cases of non-convergence
    limit = 50 # should be more than sufficient.

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
def rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor):
    m = theta.shape[0]
    
    # b: shape (m, Y_dim)
    b = np.zeros((m, Y_dim))
    b[m-1, 0] = theta[-1] * kappa[-1]

    # beta: shape (m, m)
    beta = np.zeros((m, m))
    for i in range(m):
        beta[i, i] = -kappa[i]
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

    # Build and return the struct
    # lhc = LHCStruct(a, c, gamma, b, beta,
    #                 A, A_star, A_star_inv,
    #                 id_mat, r, m, Y_dim,
    #                 delta, tenor)
    lhc_tuple = (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor)

    return lhc_tuple


@njit
def rebuild_lhc_struct_fast(lhc_old, kappa, theta, gamma1, r, Y_dim, delta, tenor):
    m = len(theta)
    b = np.zeros((m, Y_dim))
    b[m-1, 0] = theta[-1] * kappa[-1]

    beta = np.zeros((m, m))
    for i in range(m):
        beta[i, i] = -kappa[i]
        if i + 1 < m:
            beta[i, i+1] = kappa[i] * theta[i]

    gamma = np.zeros((1, m))
    gamma[0, 0] = -gamma1
    c = np.zeros((Y_dim, Y_dim))

    A = np.zeros((Y_dim + m, Y_dim + m))
    A[:Y_dim, :Y_dim] = c
    A[:Y_dim, Y_dim:] = gamma
    A[Y_dim:, :Y_dim] = b
    A[Y_dim:, Y_dim:] = beta

    id_mat = np.eye(Y_dim + m)
    A_star = A - r * id_mat

    # det_A_star = np.linalg.det(A_star)
    # if (np.isnan(det_A_star))| (np.abs(det_A_star)<1e-12) :
    #     A_star_inv = np.linalg.pinv(A_star)
    # else:
    A_star_inv = np.linalg.inv(A_star)

    a = np.ones((Y_dim, 1))

    lhc = (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor)
    return lhc

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
    t_grid_len = int(np.round((t_M - t0) / tenor).item()) + 1
    t_grid = np.zeros(t_grid_len)
    for i in range(t_grid_len):
        t_grid[i] = t0 + i * tenor
    for j in range(1, t_grid_len):
        dt = t_grid[j] - t_grid[j-1]
        sum_Z += dt * psi_Z(lhc, t, t_grid[j])
        if j < t_grid_len - 1:
            sum_D += dt * psi_D(lhc, t, t_grid[j])
    return (sum_Z + psi_D_star(lhc, t, t_M) - psi_D_star(lhc, t, t0)
            + t_grid[-2] * psi_D(lhc, t, t_M) - sum_D - t0 * psi_D(lhc, t, t0))

@njit
def psi_cds(lhc, t, t0, t_M, k):
    return psi_prot(lhc, t, t0, t_M) - k * psi_prem(lhc, t, t0, t_M)

@njit
def get_CDS_Model(t_obs, t0, t_mat_grid, state_vec, lhc):
    n_mat, n_obs = t_mat_grid.shape
    CDS = np.ones((n_mat, n_obs))
    for mat_idx in range(n_mat):
        for i in range(n_obs):
            prot = psi_prot(lhc, t_obs[i], t0[i], t_mat_grid[mat_idx, i])
            prem = psi_prem(lhc, t_obs[i], t0[i], t_mat_grid[mat_idx, i])
            st = state_vec[:, i]
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
    result = np.zeros((t_mat_grid.shape[0],m+Y_dim), dtype=np.float64)
    for i in range(t_mat_grid.shape[0]):
        # Pass a scalar from the array X
        prem = psi_prem(lhc,t,t0,t_mat_grid[i])
        prot = psi_prot(lhc,t,t0,t_mat_grid[i])
        term1 = prot / np.dot(prem, chi)
        term2 = np.dot(prot, chi) / np.dot(prem, chi)**2 *prem
        result[i,:] = term1 - term2
        
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
def update_step_cds(X_pred, P_pred, h,h_p, R_k, t_obs,t0, t_mats, lhc, CDS_k):
    # pred_Xn = np.append([1],X_pred) # Add one for computations of cds spread and derivative
    # Step 3: Mean prediction, covariance, Kalman Gain etc.
    mu_k = h(lhc, X_pred, t_obs,t0, t_mats) 
    
    H_x = h_p(lhc, X_pred, t_obs,t0, t_mats)
    # covariance
    S_k = H_x @ P_pred @ H_x.T + R_k
    det_S = np.linalg.det(S_k)
    if  (np.isnan(det_S)) | (np.abs(det_S) < 1e-12 ):
        S_k_inv = np.linalg.pinv(S_k)
    else:
        S_k_inv = np.linalg.inv(S_k)

    # Step 4: Compute Kalman Gain, filtered mean state, covariance.
    K_k = P_pred @ H_x.T @ S_k_inv
    vn = (CDS_k - mu_k) # In Linear Approx instance
    m_k = X_pred + K_k @ vn
    P_k = P_pred - K_k @ S_k @ K_k.T

    return mu_k, vn,S_k, m_k, P_k


@njit
def build_P_params(params,theta,gamma1,lhc_p):
        (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor) = lhc_p 
        kappa = params[:m]
        sigma_i,sigma_err =params[m:2*m], params[-1]

        # Rebuild P parameters.
        lhc = rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor)


        return lhc,kappa, sigma_i, sigma_err



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
    B = beta
    
    # NOTE: Parametrization. Need to use gamma^p=-gamma
    g = - gamma.flatten()   # ensure 1D
    n = x.size

    s = g @ x                    # scalar g^T x
    outer_xg = np.outer(x, g)    # x g^T

    J = np.eye(n) + Delta * (B - (s * np.eye(n) + outer_xg))
    return J




@njit 
def matrix_sqrt(A):
    # Thanks to https://stackoverflow.com/questions/71232192/efficient-matrix-square-root-of-large-symmetric-positive-semidefinite-matrix-in
    D, V = np.linalg.eig(A)
    Bs = (V * np.sqrt(D)) @ V.T
    return Bs


### NOTE: THIS METHODOLODY WILL NOT WORK, STILL NOT OPTIMIZING AT EACH STEP (SO USE PREVIOUS VAL)
@njit
def get_states(lhc, t_obs, T_M_grid, CDS_obs,X0):
    (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor) = lhc 
    # RETHINK THIS A LOT. SEEMS LIKELY THAT THERE IS SOME SORT OF ERROR HERE. 
    n_obs = len(t_obs)
    n_mat = T_M_grid.shape[0]

    # Define initial values
    X0 = np.ones(shape=(m,)) *X0
    X = np.ones((m, n_obs))
    Y = np.ones((n_obs)) # Implicitly sets Y0
    # Time 0 values
    X[:, 0] = X0
    # Previous Z, starting guess
    Z = np.ones((m,n_obs))
    Y_prev = Y[0]
    X_prev = X[:,0]
    Z_prev = X[:,0] / Y[0]


    ti = t_obs[1]
    ti_prev = t_obs[0]
    dt = ti-ti_prev
    # Find Z,X,Y
    for time_idx in range(0, n_obs):
        A_big = np.empty(shape = (n_mat,m))
        y_big = np.empty(shape = (n_mat))
        # Weight matrix
        W = np.zeros(shape=(n_mat,n_mat))
        ti = t_obs[time_idx]
        # Build stacked vector
        one_Z = np.empty(shape = (1 + m))
        one_Z[0] = 1.0
        one_Z[1:] = Z_prev.flatten()
        
        for mat_idx in range(n_mat):
            psi_c = psi_cds(lhc,ti, ti, T_M_grid[mat_idx,time_idx], CDS_obs[time_idx,mat_idx])
            psi_p = psi_prem(lhc,ti, ti, T_M_grid[mat_idx,time_idx])
            d_k = np.dot(psi_p, one_Z)
            A_big[mat_idx,:] = - psi_c[1:] 
            y_big[mat_idx] =  psi_c[0]      # Note, y needs to be negative to formulate as WLS problem
            W[mat_idx,mat_idx] = 1 / d_k**2      # Needs to be squared to match reg .
        
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
            Y[time_idx]=1 
        else:
            Y[time_idx] = Y_prev + dt * (gamma.flatten() @ X_prev)
        
        X[:,time_idx] = Y[time_idx] * Z[:,time_idx]


        # Bump previous value
        Z_prev = Z[:,time_idx]
        X_prev = X[:,time_idx]
        Y_prev = Y[time_idx]


    return X,Y,Z


@njit
def solve_mu1(kappa, theta, gamma1):
    m = len(kappa)

    # Define f(mu1)
    def f(mu1):
        prod = 1.0
        for j in range(m):
            prod *= kappa[j] * theta[j] / (mu1 * gamma1 - kappa[j])
        return ((-1)**m) * prod - mu1

    # Bisection method parameters
    a = 1e-6
    b = 1.0 - 1e-6
    fa = f(a)
    fb = f(b)

    # Check for valid sign change
    if fa * fb > 0 or not np.isfinite(fa) or not np.isfinite(fb):
        # fallback: safe clipped mean value
        return max(min(0.5, 0.99), 0.01)

    # Bisection loop
    for _ in range(100):
        mid = 0.5 * (a + b)
        fm = f(mid)
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
def compute_stationary(kappa, theta, m, gamma1, mu1):
    mu_process = np.zeros(m)
    for i in range(m-1, -1, -1):
        sign = (-1)**(m - (i+1) + 1)
        prod = 1.0
        for j in range(i, m):
            prod *= kappa[j] * theta[j] / (mu1 * gamma1 - kappa[j])
        mu_process[i] = sign * prod
    return mu_process


# One kalman filter for optimizing and one for outputting.
@njit
def kalmanfilter_opt(params, t_obs,t0,T_M_grid,CDS_obs,lhc_p,lhc_q,X0,G_n,p_idx_matrix):
    (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor) = lhc_q 
    (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor) = lhc_p 
    # Define the parameters already to be able to look over them
    gamma1 = params[2*m]
    params_p = params[2*m+1:]
    kappa_p, sigma, sigma_err = params_p[:m],params_p[m:2*m],params_p[-1]
    # lhc_p,kappa_p,theta_p, sigma, sigma_err = build_P_params(params_p,gamma1,lhc_p)
    params_q = params[:2*m+1]
    kappa, theta, gamma1 = params_q[:m],params_q[m:2*m], params_q[-1]

    # Get initial guesses.
    n_obs = CDS_obs.shape[0]
    n_mat = CDS_obs.shape[1]

    Delta = t_obs[1] - t_obs[0] # Only apprx for now. Move to loop maybe.

    # Only A_trans utilzes new params
    _,Sigma,R_k = build_matrices(lhc_p,sigma,sigma_err,n_mat)

    L = int(m+Y_dim)
    log_likelihood = 0

    # Just to set Xn,Pn, but not needed to be thse vals.
    # Don't know these values. Just arbitrary guessing. on X. Remainder calc.
    Y0 = np.array([1])
    X0 = np.ones(shape=(m,)) * X0
    Z0 = np.ones(Y0.size+X0.size, dtype=np.float64)
    Z0[Y0.size:] = X0.ravel()/ Y0

    # Store predictions. 
    pred_Xn = np.zeros(L)
    pred_Pn = np.zeros((X0.shape[0],X0.shape[0]))
    Xn = np.zeros((n_obs,L))
    Zn = np.zeros((n_obs,n_mat))
    Pn = np.zeros((n_obs,L,L))
    # Initial Predictions of means and cov
    # Only criteria on mu1 - mu1 < k/kappa. Set to theta
    Y0 = 1
    mu1 = solve_mu1(kappa_p, theta, gamma1)
    mu = compute_stationary(kappa_p,theta,m,gamma1,mu1= mu1)
    # Note, mu corresponds to X initially.
    pred_Xn = np.append([1],mu) #drift_term(mu,lhc_p,Delta) #mu
    # Just set the covariance guess to the innovated one.
    P_state = np.array([mu[i] * (1 - mu[i]) for i in range(0,mu.shape[0]) ])
    # Initial Cov Prediction.Z0
    # Sigma = Sigma[1:,:]
    Sigma_prod = (Sigma @ np.diag(np.sqrt(P_state))) @ (Sigma @ np.diag(np.sqrt(P_state))).T 
    # Just an attemt, not to keep.
    P0 = Sigma_prod.copy()
    #pred_Pn = P0[1:,1:] 
    pred_Pn = P0

    ## Asuume same time point dist so not needed to compute every time.
    A_kalman = mat_exp_approx(A,Delta)

    G_prod = precompute_G_products(G_n, p_idx_matrix, Delta)
    
    # Run algo. 
    for n in range(0,n_obs):
        # Extended kalman filter.
        _, vn,S_k, Xn[n,:], Pn[n,:,:] = update_step_cds(pred_Xn,pred_Pn,cds_fun,cds_deriv,R_k,
                                                            t_obs[n],t0[n],T_M_grid[:,n],lhc_q,CDS_obs[n,:])
        # Compute Zn too
        # Xn_extended = np.append([1],Xn[n,:])
        Zn[n,:] =  cds_fun(lhc_q,Xn[n,:],t_obs[n],t0[n],T_M_grid[:,n])

        # add additional check until parameters are figured out
        # This is mainly an issue in local optimizers. 
        if (np.any(Xn<0) |np.any(Xn>1)):
            return 1e12,Xn, Zn, Pn 

        # Update log likelihood.            
        det_S = np.linalg.det(S_k)
        if det_S < 0:
            return 1e12,Xn, Zn, Pn 

        # Some fallback / numerical fixes
        if (np.isnan(det_S)) | (det_S < 1e-12) :
            S_inv = np.linalg.pinv(S_k)
        else:
            S_inv = np.linalg.inv(S_k) 
        
        # Harsh penalty if too unstable determinant.
        # if (np.isnan(det_S)) | (det_S < 1e-12):
        #     return 1e12,Xn, Zn, Pn 
        ll_step = -0.5 * (S_k.shape[0] * np.log(2*np.pi) + np.log(det_S) + vn.T @ S_inv @ vn)

        log_likelihood += ll_step

        if (n < n_obs - 1): # Not sensible to predict further.
            Delta = t_obs[n+1] - t_obs[n] # Only apprx for now. Move to loop maybe.
            
            # Pure Euler
            pred_Xn = A_kalman @ Xn[n,:]
            # New Q_t is poly cov.
            basis = monomial_basis_numba(Xn[n,0],Xn[n,1:],poly_deg=2)
            Q_k = compute_second_moment_from_basis(basis,G_prod) -np.outer(pred_Xn,pred_Xn)
            pred_Pn =  A_kalman @ Pn[n,:,:] @ A_kalman.T + Q_k

    return - log_likelihood, Xn, Zn, Pn 

# @njit
def kalman_wrapper(params, t_obs,t0,T_M_grid,CDS_obs,X0,m,r, Y_dim, delta, tenor):
    # For numerical stability.
    params_p = params[2*m+1:]
    params_q = params[:2*m+1]
    kappa_p = params_p[:m]
    kappa, theta, gamma1 = params_q[:m],params_q[m:2*m], params_q[-1]
    lhc_p = rebuild_lhc_struct(kappa_p, theta, gamma1, r, Y_dim, delta, tenor)
    lhc_q = rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor)

    ## Compute conditional moments...
    # only secon moments
    poly_deg = 2
    N_n = math.comb(poly_deg+1+m,poly_deg)
    basis,index = monomial_basis(1,np.ones(shape=m),poly_deg)
    G_n = poly_G(m,lhc_p,params_p[m:-1],poly_deg,index)

    # Get coordinate vectors. 
    names = ['y'] + [f'X{i+1}' for i in range(m)]
    p_idx_matrix = np.zeros((1+m, 1+m), dtype=np.int64)

    for i in range(1+m):
        for j in range(i, 1+m):
            # map to monomial index (example: yX1 -> 5)
            key = i == 0 and 'y' or f'X{i}'
            key2 = j == 0 and 'y' or f'X{j}'
            # ordering: smaller index first for consistency
            key_str = key + key2 if i <= j else key2 + key
            p_idx_matrix[i,j] = index[key_str]

    # kalman filter
    print(params)
    neg_loglik,_, _, _ = kalmanfilter_opt(params, t_obs,t0,T_M_grid,CDS_obs,lhc_p,lhc_q,X0,G_n,p_idx_matrix)

    return neg_loglik

def nonlinear_constraints( params, m):
    """
    Vector-valued constraint function for NonlinearConstraint.
    Returns array `cons` such that cons >= 0 are feasible.
    If params can't be unpacked properly, returns a negative array (infeasible).
    """
    m = int(m)
    # expected number of constraints: g1, g2 and 2*m for g3,g4 converted -> total 2 + 2*m
    params = np.asarray(params, dtype=float)

    params_p = params[2*m+1:]
    params_q = params[:2*m+1]
    kappa_p,sigma,sigma_err = params_p[:m],params_p[m:2*m],params_p[-1]
    kappa, theta, gamma1 = params_q[:m],params_q[m:2*m], params_q[-1]
    # ensure arrays
    kappa_p = np.asarray(kappa_p, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Build constraints: all should be >= 0
    cons = []

    # g1 >= 0  -> theta[-1]*kappa[-1] - sigma[-1]^2/2 >= 0
    for i in range(m):
        g1 = theta[i] * kappa[i] - 0.5 * (sigma[i] ** 2)
        cons.append(g1)

    # g2 >= 0  -> theta_p[-1] * kappa_p[-1] - sigma[-1]^2/2 >= 0
    for i in range(m):
        g2 = theta[i] * kappa_p[i] - 0.5 * (sigma[i] ** 2)
        cons.append(g2)
    # g3 <= 0 -> -g3 >= 0
    for i in range(m):
            g3 = gamma1 - kappa[i] + kappa[i] * theta[i] + 0.5 * (sigma[i] ** 2)
            cons.append(-g3)


    # g4 <= 0 -> -g4 >= 0
    for i in range(m):
            g4 = gamma1 - kappa_p[i] + kappa_p[i] * theta[i] + 0.5 * (sigma[i] ** 2)
            cons.append(-g4)

    cons = np.asarray(cons, dtype=float)


    return cons


class LHC_single():
    def __init__(self, r, delta, cds_tenor):
        # Set global params 
        self.r = r                  # Set short rate
        self.delta = delta          # Set recovery rate
        self.tenor = cds_tenor      # Set Swap tenor/payments structure.

    
    def initialise_LHC(self, Y_dim, X_dim,X0, rng=None):
        if rng is None:
            rng = np.random.default_rng()  # independent each time

        self.Y_dim, self.m = Y_dim, X_dim
        self.X0 = X0
        self.a = np.ones((self.Y_dim,1))                                      # Y dim is 1 for LHC
        # Set inital values. Need to comply with (38)
        self.gamma1 = rng.uniform(0.01, 0.6, size=(Y_dim,))       # gamma1 strictly pos.
        self.kappa = rng.uniform(self.gamma1, 1.1, size=(X_dim,))       # Kappa given 
        # self.gamma1 = rng.uniform(0.05, 0.3, size=(Y_dim,))       # gamma1 strictly pos.
        
        self.theta = np.zeros(X_dim)

        for i in range(0,X_dim):
            self.theta[i] = rng.uniform(0, 1-self.gamma1/self.kappa[i], size=(1,))       # Theta coeffs
        # Build b, beta, A, gamma
        self.rebuild_dynamics()                                     # Build b,beta,gamma again.


    def rebuild_dynamics(self):
        # Formulas, cf p. 16.
        self.b = np.zeros(len(self.theta)).reshape((self.m, self.Y_dim))
        self.b[-1,:] = self.theta[-1] * self.kappa[-1]
        self.beta = np.zeros((self.m, self.m))
        for i in range(0,self.m):
            self.beta[i, i] = - self.kappa[i]
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
        return (np.exp(-self.r * (t_M - t)) * a_zeros @ expm(self.A * (t_M - t))).ravel()

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
        return (sum_Z + self.psi_D_star( t, t_M) - self.psi_D_star(t, t0)
                + t_grid[-2] * self.psi_D(t, t_M) - sum_D - t0 * self.psi_D( t, t0))

    def psi_cds(self, t, t0, t_M, k):
        return self.psi_prot(t, t0, t_M) - k * self.psi_prem(t, t0, t_M)


    def CDS_model(self,t_obs, T_M_grid, CDS_obs, t0=None, X_in=None,Y_in=None,Z_in=None):
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
            X_in,Y_in, Z = get_states(lhc, t_obs, T_M_grid, CDS_obs,self.X0)

        if Z_in is not None:
            X_in,Y_in = self.kalman_X_Y(t_obs,Z_in)

        X,Y = X_in,Y_in

        #print('Done Getting Z,Y,X')
        state_vec = np.vstack([Y, X])

        # Here formula is the t_obs according to formula
        CDS = get_CDS_Model(t_obs, t_obs, T_M_grid, state_vec, lhc )
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
        X,Y,Z = get_states(lhc, t_obs, T_M_grid, CDS_obs,self.X0)

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
    def objective(self, params, t_obs, T_M_grid, CDS_obs):
        # Test for feasibility.

        # --------- HARD CONSTRAINT CHECKS ---------
        # 1. Positivity
        if np.any(params <= 0):
            return 1e12  # infeasible, huge loss

        # 2. Custom constraints
        g1 = params[self.m-1] * params[2*self.m-1]
        if g1 < 0:
            return 1e12

        for i in range(self.m):
            g3 = params[2*self.m] - params[i] + params[i] * params[self.m+i]
            if g3 > 0:   
                return 1e12


        # Format params for calculations
        # self.test_constriants()
        self.unflatten_params(params)


        #  Build Psi functions to avoid redoing it later.
        model_cds = self.CDS_model(t_obs, T_M_grid, CDS_obs)
        obj = np.sqrt(np.mean((model_cds - CDS_obs)**2))
        
        return obj


    def optimize_params(self,t_obs, T_M_grid, CDS_obs):
        # Retrieve initial parameters. 
        flat_init = self.flatten_params().copy()
        CDS_obs = np.ascontiguousarray(CDS_obs)

        result = minimize(
            fun = self.objective,
            x0 = flat_init,
            method='Nelder-Mead',
            # method = 'L-BFGS-B', # Finite difference method.
            args = (t_obs, T_M_grid, CDS_obs),
            options = {
                "xatol": 1e-4,
                "fatol": 1e-4,
                "maxiter": 500,
                "disp": True
            }
        )
        # constraints = self.build_constraints(self.m)



        if result.success:
            print(f"Optimization succeeded, params:{result.x}, objective: {result.fun}")
            self.unflatten_params(result.x)
            self.objective_result = result.fun
        else:
            print("Optimization failed:", result.message)
            self.unflatten_params(result.x)
            self.objective_result = result.fun

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
    def build_P_params(self,params=None,theta=None, gamma1=None,rng=None):
        # NOTE: Gamma does not change!
        if params is None:
            if rng is None:
                rng = np.random.default_rng()  # independent each time
            Y_dim, X_dim = self.Y_dim, self.m
            gamma1 = self.gamma1[0]
            theta = self.theta
            # Set inital values. Need to comply with (38)
            kappa = rng.uniform(gamma1, 0.99, size=(X_dim,))       # Kappa given 
            # theta = np.zeros(X_dim)

            # for i in range(0,self.m):
            #     theta[i] = rng.uniform(1e-6, 1-gamma1/kappa[i], size=(1,))       # Theta coeffs
            
            ### New stuff: All the ones needed here e.g. sigma, sigma_Err
            sigma_i = rng.uniform(0.1, 0.7, size=(X_dim,))       # Kappa given 
            # Sigma error is likely smalll.
            sigma_err = rng.uniform(0.001, 0.01, size=(Y_dim,))       # Kappa given 

            r = self.r
            Y_dim = self.Y_dim
            delta = self.delta
            tenor = self.tenor 
            # Rebuild P parameters.
            lhc = rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor)

        else:
            gamma1 = gamma1[0] # due to parametrization
            kappa = params[:self.m]
            sigma_i,sigma_err =params[self.m:2*self.m], np.array([params[-1]])
            r = self.r
            Y_dim = self.Y_dim
            delta = self.delta
            tenor = self.tenor 
            # Rebuild P parameters.
            lhc = rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor)

        self.kappa_p, self.sigma, self.sigma_err=kappa, sigma_i, sigma_err

        return lhc

    # Get notmalized drifft stuff.
    def solve_mu1(self,kappa, theta, gamma1):
        m = len(kappa)

        def f(mu1):
            prod = 1.0
            for j in range(m):
                prod *= kappa[j] * theta[j] / (mu1 * gamma1 - kappa[j])
            return ((-1)**m) * prod - mu1

        # Solve only for admissible range
        return brentq(f, 1e-6, 1.0-1e-6)

    def compute_stationary(self,kappa, theta, m, gamma1, mu1):
        mu_process = np.zeros(m)
        for i in range(m-1, -1, -1):
            sign = (-1)**(m - (i+1) + 1)
            prod = 1.0
            for j in range(i, m):
                prod *= kappa[j] * theta[j] / (mu1 * gamma1 - kappa[j])
            mu_process[i] = sign * prod
        return mu_process


    # One kalman filter for optimizing and one for outputting.

    def get_kalman_params(self, t_obs, T_M_grid, CDS_obs, x0, lhc_p,lhc_q,base_seed):
        (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor)=lhc_p
        t0 = t_obs
        bounds = (
            [(1e-6, 1)] * m +     # kappa
            [(1e-6, 1)] * m +     # theta
            [(0.001,1)] +        # gamma1
            [(1e-6, 1)] * m +     # kappa_p
            [(1e-6, 1)] * m +     # sigma 
            [(1e-6, 0.5)]         # sigma_err
        )

        
        nlc = NonlinearConstraint(lambda x: nonlinear_constraints(x,self.m), 0, np.inf)

        # Global optimizer: few iters. Do on supsed only. 
        result = differential_evolution(
            func=kalman_wrapper,
            x0=x0,
            bounds=bounds,
            constraints=(nlc,),
            # args=(t_obs[::5], t0[::5], T_M_grid[:, ::5], CDS_obs[::5, :],
            #        self.X0,self.m,self.r, self.Y_dim, self.delta, self.tenor),
            args=(t_obs, t0, T_M_grid, CDS_obs,
                self.X0,self.m,self.r, self.Y_dim, self.delta, self.tenor),
            strategy='best1bin',
            popsize=3,         # larger popsize -> more exploration
            maxiter=100,       # allow many generations          
            tol=1e-2,            # looser tolerance
            # workers=1,
            # updating='immediate',
            workers=-1,
            updating='deferred',
            polish=False,
            seed=base_seed
        )

        # If DE failed or hit constraint penalties, bail early
        if result.fun >= 1e12:
            return result.x, 0, 0, 0


        # Flatten the list of tuples into lower and upper arrays
        # lower = [b[0] for b in bounds]
        # upper = [b[1] for b in bounds]

        # bounds_tc = Bounds(lower, upper)

        # # Nonlinear constraint stays exactly the same
        # nlc = NonlinearConstraint(lambda x: nonlinear_constraints(x, self.m), 0, np.inf)
        # x0 = result.x
        # local_result = minimize(
        #     fun=kalman_wrapper,
        #     x0=x0,
        #     args = (t_obs, t0, T_M_grid, CDS_obs,
        #             self.X0,self.m,self.r, self.Y_dim, self.delta, self.tenor),     
        #     method='trust-constr',          # supports nonlinear constraints
        #     bounds=bounds_tc,
        #     constraints=[nlc],
        #     options={'maxiter': 200, 'gtol': 1e-6, 'verbose': 2}
        # )
        # optim_params = local_result.x
        # self.kalman_obj = local_result.fun

        optim_params = result.x
        self.kalman_obj = result.fun

        # --- Evaluate Kalman filter at final parameters ---
        if self.kalman_obj >= 1e12:
            return optim_params, 0, 0, 0
        else:
            params_p = optim_params[2*m+1:]
            params_q = optim_params[:2*m+1]
            kappa_p = params_p[:m]
            kappa, theta, gamma1 = params_q[:m],params_q[m:2*m], params_q[-1]
            lhc_p = rebuild_lhc_struct(kappa_p, theta, gamma1, r, Y_dim, delta, tenor)
            lhc_q = rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor)

            ## Compute conditional moments...
            # only secon moments
            poly_deg = 2
            N_n = math.comb(poly_deg+1+m,poly_deg)
            basis,index = monomial_basis(1,np.ones(shape=m),poly_deg)
            G_n = poly_G(m,lhc_p,params_p[m:-1],poly_deg,index)

            # Get coordinate vectors. 
            names = ['y'] + [f'X{i+1}' for i in range(m)]
            p_idx_matrix = np.zeros((1+m, 1+m), dtype=np.int64)

            for i in range(1+m):
                for j in range(i, 1+m):
                    # map to monomial index (example: yX1 -> 5)
                    key = i == 0 and 'y' or f'X{i}'
                    key2 = j == 0 and 'y' or f'X{j}'
                    # ordering: smaller index first for consistency
                    key_str = key + key2 if i <= j else key2 + key
                    p_idx_matrix[i,j] = index[key_str]

            # kalman filter
            neg_loglik, Xn, Zn, Pn = kalmanfilter_opt(optim_params, t_obs,t0,T_M_grid,CDS_obs,lhc_p,lhc_q,self.X0,G_n,p_idx_matrix)

            # Build structures in very end to take onwards.
            self.build_P_params(optim_params,theta,np.array([gamma1]))
            self.flatten_params()
            self.unflatten_params(params_q)
            
            return optim_params, Xn, Zn, Pn
        
    def run_n_kalmans(self, t_obs,T_M_grid, CDS_obs, base_seed = 1000,  n_restarts = 20):
        # Define grid of values. 
        current_objective = 1e10 #very high objective.
        out_params = self.flatten_params()
        for i in range(n_restarts):
            print(f"Optimization {i+1}")
            rng = np.random.default_rng(base_seed + i)  # deterministic but different
            # Set Q parameters.
            self.initialise_LHC(self.Y_dim,self.m,self.X0,rng)
            # Get P Parameters /initialise
            lhc_p = self.build_P_params(params=None, theta=None, gamma1=None,rng=rng)
            
            # Set the error to be the Stddeviation of CDS_obs
            #self.sigma_err = np.std(CDS_obs).flatten()
            # Flatten for scipy. 
            x0_Q = self.flatten_params()

            x0_P = np.concatenate([
                self.kappa_p.flatten(),
                self.sigma.flatten(),
                self.sigma_err
            ])

            x0 = np.concatenate([x0_Q,x0_P])
            (a, c, gamma, b, beta,
                                A, A_star, A_star_inv,
                                id_mat, r, m, Y_dim,
                                delta, tenor) = lhc_p
            gamma1 = x0[2*m]
            params_p = x0[2*m+1:]
            lhc_p,kappa_p, sigma, sigma_err = build_P_params(params_p,self.theta,gamma1,lhc_p)
            params_q = x0[:2*m+1]
            kappa, theta, gamma1 = params_q[:m],params_q[m:2*m], params_q[-1]
            # build Q class outside wrapper too:
            lhc_q = rebuild_lhc_struct(kappa,theta,gamma1,self.r,self.Y_dim,self.delta,self.tenor)

            # Test several random points. 
            optim_params,  Xn,Zn, Pn= self.get_kalman_params(t_obs,T_M_grid, CDS_obs,x0,lhc_p,
                                                             lhc_q, base_seed=base_seed + i)
            # Test new constraints

            # if (self.kalman_obj < current_objective):
            #     print(f"New optimal parameters at iteration {i+1}.")
            #     current_objective = self.kalman_obj
            #     out_params, Xn_out,Zn_out, Pn_out = optim_params, Xn,Zn, Pn
            #     self.unflatten_params(out_params[:x0_Q.shape[0]])

        # Set new optimal parameters. 
        return  optim_params,  Xn,Zn, Pn # out_params,  Xn_out,Zn_out, Pn_out

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
    def simul_latent_states(self, chi0, T,M,n_mat,seed=None,scheme='Euler'):
        delta = T / M
        T_return = np.array([0] + [delta*k for k in range(1,M+1)])
        path_Q = np.ones((M + 1, self.m+self.Y_dim))

        # Set initial value. 
        path_Q[0,:] = chi0
        W_Q = norm.rvs(size = (M,self.m),random_state=seed) # simulate at beginning - faster!

        # Get A matrix. Add argument if timul under P or Q.
        params_Q = np.concatenate([self.kappa,self.theta,self.gamma1])
        # If params have been sat with meaningfull values
        params_P = np.concatenate([self.kappa_p, self.sigma, self.sigma_err])
        # If explicitly given, manual sigma is used.
        params = np.concatenate([params_Q,params_P])
        # Set also P params. 
        lhc_p = self.build_P_params(params_P,self.theta, self.gamma1)
        _,cov_trans,_ = build_matrices(lhc_p,self.sigma,self.sigma_err,n_mat)
    
        self.rebuild_dynamics()
        # Get A
        A = self.A
        if scheme == 'Euler':
            for i in range(1,M+1):
                mu_t = A @ path_Q[i - 1,:]
                # Create Sigma:
                P_state = np.array(path_Q[i - 1,1:] * (path_Q[i-1,0] - path_Q[i-1,1:]))
                Sigma_prod = (cov_trans @ np.diag(np.sqrt(P_state))) 
                # Out
                path_Q[i,:] = path_Q[i-1,:] + delta*mu_t +  np.sqrt(delta) * Sigma_prod @ W_Q[i-1,:]
        
        if scheme == 'Milstein':
            for i in range(1,M+1):
                mu_t = A @ path_Q[i - 1,:]
                # Create Sigma:
                P_state = np.array(path_Q[i - 1,1:] * (path_Q[i-1,0] - path_Q[i-1,1:]))
                Sigma_prod = (cov_trans[1:,: ]   @ np.diag(np.sqrt(P_state))) 
                P_state_mil =  np.diag(np.array(1/(2*np.sqrt(path_Q[i - 1,1:] * (path_Q[i-1,0] - path_Q[i-1,1:])))
                                       * (path_Q[i-1,0] - 2*path_Q[i-1,1:])))

                sigma_prime_t = cov_trans[1:,: ]  @ P_state_mil 

                # sigma to mult on BM
                sigma_with0 = np.vstack((np.zeros(self.m), Sigma_prod))
                sigma_mil_with0 = np.vstack((np.zeros(self.m),sigma_prime_t @ Sigma_prod))

                # Out
                path_Q[i,:] = (path_Q[i-1,:] + delta*mu_t +  
                               np.sqrt(delta) * sigma_with0 @ W_Q[i-1,:] + 
                               1/2 * delta* sigma_mil_with0 @ (W_Q[i-1,:]**2-1))
        
        return T_return, path_Q
    

    # Simulate option prices in the model. 
    def get_cdso_pric_MC(self,t,t0,t_M,strikes,chi0,N,M,seed=1000, P_params=None):
        # If p params specific (Sigma specific), se can calculate for differnt sigma
        if P_params is not None:
            lhc_p = self.build_P_params(P_params,self.theta,self.gamma1)

        # Get LHC_Q for pricing
        lhc = rebuild_lhc_struct(self.kappa, self.theta, self.gamma1[0], self.r,
                            self.Y_dim, self.delta, self.tenor)

        # N prices are comuted and averaged MC
        N_strikes = strikes.shape[0]
        prices = np.zeros(shape = (N,N_strikes))
        prices_MC_hist = np.zeros(shape = (N,N_strikes))
        for i in range(N):
            # Get Latent states. Simulate to time of inception of CDS. Calculate only for 1 maturity.
            T_return, X_Q = self.simul_latent_states( chi0,t0,M,n_mat=1,seed=seed)
            S = X_Q[:,0]
            # Determine if default or not at t0. If S_t \leq unif(1) option payoff is zero.
            U = uniform.rvs(random_state=seed)
            # If survival falls below U at any points, default happened prior to t0
            if np.any(S <= U):
                prices[i] = 0
            # Else - begin to compute prices as no default
            else: 
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
            lhc_p = self.build_P_params(P_params,self.theta,self.gamma1)

        # N prices are comuted and averaged MC
        N_strikes = barriers.shape[0]
        prices = np.zeros(shape = (N,N_strikes))
        prices_MC_hist = np.zeros(shape = (N,N_strikes))
        lhc = rebuild_lhc_struct(self.kappa, self.theta, self.gamma1[0], self.r,
                            self.Y_dim, self.delta, self.tenor)
        for i in range(N):
            # Get Latent states. Simulate path of CDS till mat.
            T_return, X_Q = self.simul_latent_states( chi0,T,M,n_mat=1,seed=seed)
            Xn = X_Q[:,self.Y_dim:]
            Yn = X_Q[:,0]
            # Retrieve price path...
            T_M_grid = ((T_return + t_M)).reshape((1,T_return.shape[0]))
            # Here formula is the t_obs according to formula
            CDS_sim = get_CDS_Model(T_return, T_return+t0, T_M_grid, X_Q.T, lhc )
            S = Yn
            # Determine if default or not at t0. If S_t \leq unif(1) option payoff is zero.
            U = uniform.rvs(random_state = seed)
            # If survival falls below U at any points, default happened prior to T. 
            # Should be zero, but depends on barrier. Default happens at some point below..
            default_event =  S <= U
            if np.any(default_event):
                # In this instance, default has happened as some point. Find index. 
                idx = np.argmax(np.where(default_event))
                # Get path maximum up to the point:
                max_cds_to_default = np.max(CDS_sim[:idx+1])
                for b_idx in range(N_strikes):
                    if max_cds_to_default >= barriers[b_idx]:
                        # In this case, there is a payoff of 1, discount back from expiry(pay date) to today
                        prices[i,b_idx] = np.exp(-self.r * (T - t))
                    else:
                        # If not above, there is zero payoff.
                        prices[i,b_idx] = 0
            # Else - no default happened, but same logic as before.
            else: 
                # Get path maximum up to expiry:
                max_cds_to_default = np.max(CDS_sim)
                for b_idx in range(N_strikes):
                    if max_cds_to_default >= barriers[b_idx]:
                        # In this case, there is a payoff of 1, discount back from expiry(pay date) to today
                        prices[i,b_idx] = np.exp(-self.r * (T - t))
                    else:
                        # If not above, there is zero payoff.
                        prices[i,b_idx] = 0
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
            lhc_p = self.build_P_params(P_params,self.theta,self.gamma1)
        lhc = rebuild_lhc_struct(self.kappa, self.theta, self.gamma1[0], self.r,
                            self.Y_dim, self.delta, self.tenor)
        # N prices are comuted and averaged MC
        prices = np.zeros(shape = N)
        for i in range(N):
            # Get Latent states. Simulate path of CDS till mat.
            T_return, X_Q = self.simul_latent_states( chi0,T,M,n_mat=1,seed=seed)
            # Retrieve price path...
            T_M_grid = ((T_return + t_M)).reshape((1,T_return.shape[0]))
            CDS_sim = get_CDS_Model(T_return, T_return+t0, T_M_grid, X_Q.T, lhc )
            if np.any(CDS_sim == np.nan):
                print('Nan value')

            S = X_Q[:,0]
            # Determine if default or not at t0. If S_t \leq unif(1) option payoff is zero.
            U = uniform.rvs(random_state = seed)
            # If survival falls below U at any points, default happened prior to T - No payoff
            default_event =  S <= U
            if np.any(default_event):
                prices[i] = 0
                # set min to zero here
                cds_min[i] =0  # np.min(CDS_sim[:i])
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
            lambda x: (x * GenLegendrePoly(x,n,b_min,b_max)),
            0, b_max,
            limit=200,
            epsabs=1e-12,
            epsrel=1e-12
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
            GLPoly = GenLegendrePoly(z,j,b_min,b_max)
            pi += f_j * GLPoly
        
        return pi
    



## Pricing numba functions:
@njit
def LegendrePoly(x, n):
    # Compute standard Legendre. 
    Le0, Le1 = 1,x
    if n == 0:
        return Le0
    if n == 1:
        return Le1
    else:
        n_current = 1
        Le_np1 = 0 # Just start value.
        Le_n = Le1
        Le_nm1 = Le0
        while n_current < n:
            Le_np1 = ((2*n_current + 1) * x * Le_n / (n_current + 1) -
                       n_current * Le_nm1 / (n_current+1))
            # Bump values. 
            Le_nm1 = Le_n
            Le_n = Le_np1

            # Bump current. 
            n_current += 1

        return Le_np1


@njit
def GenLegendrePoly(x, n,b_min,b_max):
    mu = (b_max + b_min) / 2
    sigma = (b_max - b_min) / 2

    mathL = np.sqrt((1+2*n)/(2*sigma**2)) * LegendrePoly((x - mu) / sigma,n)

    return mathL


##### Functionality for expectations. Likely numba for efficiency. 
    ### Compute conditional moments of Y and X. 
def monomial_basis(y,X,poly_deg):
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

@njit
def monomial_basis_numba(y,X,poly_deg):
    m = X.shape[0]
    if poly_deg == 1:
        return np.append([1,y],X)
    if poly_deg == 2:
        exp_part= np.append([1,y],X)
        second_basis = np.append(exp_part, [y**2])
        second_basis = np.append(second_basis, y*X)
        for i in range(0,m):
            second_basis = np.append(second_basis,[X[i]*X[j] for j in range(i,m)] )

    return second_basis

# Then, we are ready for logic to compute this G_n thingy. 
def poly_G(m,lhc,sigma,poly_deg,index):
    (a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor) = lhc 
    # get x dim and number of monomial basis
    N_n = math.comb(poly_deg+1+m,poly_deg)
    ### Then ready to compute. Only reasonable for n>0. 
    G_n = np.zeros(shape = (N_n,N_n))
    if poly_deg == 1:
        # Get indices to fill:
        # y parts.
        G_n[index['X1']:,index['y'] ] = gamma

        # X col part. 
        G_n[index['y']:,index['X1']: ] = np.vstack([b.T,beta.T])


    elif poly_deg == 2:
        ### First part is the same as earlier -> same rows and entries
        ##### Mean section
        # y col
        G_n[index['X1']:index['yy'],index['y'] ] = gamma
        # x col
        G_n[index['y']:index['yy'],index['X1']: index['yy']] = np.vstack([b.T,beta.T])

        ##### Second order section
        ### When Poly is y**2:
        # index y^2 col is m+1+1. 
        G_n[index['yX1']:index['X1X1'],index['yy']] =  2*gamma

        ### Polynomia yx_i i.e. Cross terms.
        # beta part...
        G_n[index['yy']:index['X1X1'],index['yX1']:index['X1X1']] =  np.concatenate([b.T,beta.T])
        # Cross term with gammas (utilize lhc form).
        gamma_flat = gamma.flatten() 

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
                G_n[index[f'yX{i}'],index[f'X{i}X{j}']] +=  b.flatten()[j-1]
                G_n[index[f'yX{j}'],index[f'X{i}X{j}']] +=  b.flatten()[i-1]
                
                # X cross contributions for all k
                for k in range(1, m+1):
                    key_i = f'X{i}X{k}' if i <= k else f'X{k}X{i}'
                    key_j = f'X{j}X{k}' if j <= k else f'X{k}X{j}'
                    if key_i in index:
                        G_n[index[key_i], index[f'X{i}X{j}']] += beta.T[ k-1,j-1]
                    if key_j in index:
                        G_n[index[key_j], index[f'X{i}X{j}']] += beta.T[k-1,i-1]

            # The double terms are what we consider now
            if i == j:
                G_n[index[f'yX{i}'],index[f'X{i}X{i}']] += sigma[i-1]**2 
                G_n[index[f'X{i}X{i}'],index[f'X{i}X{i}']] += - sigma[i-1]**2

                # Remaining terms.
                G_n[index[f'yX{i}'],index[f'X{i}X{i}']] += 2 * b.flatten()[i-1]

                for run_idx in range(1,m+1):
                    # logic to flip
                    if run_idx > i:
                        G_n[index[f'X{i}X{run_idx}'],index[f'X{i}X{i}']] += 2 *  beta.T[run_idx-1,i-1]
                    else:
                        G_n[index[f'X{run_idx}X{i}'],index[f'X{i}X{i}']] +=  2 * beta.T[run_idx-1,i-1]
                    
    return G_n

#### Calculate moments/covariance.

def moments_poly(y,X,lhc,sigma,poly_deg,p_idxs,time_delta):
    ### Get monomial basis.
    m = X.shape[0]
    N_n = math.comb(poly_deg+1+m,poly_deg)
    basis,index = monomial_basis(y,X,poly_deg)
    p_coord = np.zeros( N_n)
    p_coord[p_idxs] = 1
    G_n = poly_G(m,lhc,sigma,poly_deg,index)

    return basis @ expm(G_n * time_delta) @ p_coord


# @njit
# def compute_second_moment_numba(y,X, G_n, p_idx_matrix, time_delta):
#     basis = monomial_basis_numba(y,X,poly_deg = 2)
#     n_dim = p_idx_matrix.shape[0]
#     second_moment = np.zeros((n_dim, n_dim))
    
#     # Compute expm once (Numba-compatible)
#     G_exp = mat_exp_approx(G_n, time_delta)
    
#     # Fill upper triangle
#     for i in range(n_dim):
#         for j in range(i, n_dim):
#             p_idx = p_idx_matrix[i, j]
#             p_vec = np.zeros(G_n.shape[0])
#             p_vec[p_idx] = 1.0
#             second_moment[i, j] = basis @ G_exp @ p_vec
    
#     # Symmetrize
#     for i in range(n_dim):
#         for j in range(i):
#             second_moment[i, j] = second_moment[j, i]
    
#     return second_moment


@njit
def precompute_G_products(G_n, p_idx_matrix, time_delta):
    """Precompute exp(G_n * dt) @ p_vec for all (i, j) entries."""
    G_exp = mat_exp_approx(G_n, time_delta)
    n_dim = p_idx_matrix.shape[0]
    n_basis = G_n.shape[0]
    G_products = np.zeros((n_dim, n_dim, n_basis))

    for i in range(n_dim):
        for j in range(i, n_dim):
            p_idx = p_idx_matrix[i, j]
            p_vec = np.zeros(n_basis)
            p_vec[p_idx] = 1.0
            G_products[i, j, :] = G_exp @ p_vec
            if i != j:
                G_products[j, i, :] = G_products[i, j, :]  # symmetry
    return G_products


@njit
def compute_second_moment_from_basis(basis, G_products):
    """Compute second moment given precomputed G_products and basis."""
    n_dim = G_products.shape[0]
    second_moment = np.zeros((n_dim, n_dim))
    
    for i in range(n_dim):
        for j in range(n_dim):
            second_moment[i, j] = basis @ G_products[i, j, :]
    return second_moment
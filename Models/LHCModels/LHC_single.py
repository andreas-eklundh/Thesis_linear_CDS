import numpy as np 
from scipy.optimize import minimize, NonlinearConstraint 
from scipy.stats import norm, ncx2, gamma, expon, uniform
from scipy.integrate import solve_ivp 
from scipy.linalg import expm 
from scipy.optimize import lsq_linear 
from numba import njit, float64, int64 
from numba.experimental import jitclass 
from scipy.integrate import quad
import copy
from scipy.linalg import sqrtm
from scipy.optimize import brentq



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
    A_star_inv = np.linalg.inv(A_star) # pinv always exist, not unique

    # a vector (assume ones for simplicity)
    a = np.ones((Y_dim,1))

    # Build and return the struct
    lhc = LHCStruct(a, c, gamma, b, beta,
                    A, A_star, A_star_inv,
                    id_mat, r, m, Y_dim,
                    delta, tenor)
    return lhc

# -------- psi functions rewritten for numba -------- #
@njit
def psi_Z(lhc, t, t_M):
    dt = t_M - t
    mat_exp = mat_exp_approx(lhc.A, dt)
    a0 = np.zeros(lhc.Y_dim + lhc.m)
    a0[:lhc.Y_dim] = lhc.a.ravel()
    return np.exp(-lhc.r * dt) * (a0 @ mat_exp).ravel()

@njit
def psi_D(lhc, t, t_M):
    dt = t_M - t
    mat_exp = mat_exp_approx(lhc.A_star, dt)
    # build [c | gamma]
    c_gamma = np.zeros((lhc.Y_dim, lhc.Y_dim + lhc.m))
    c_gamma[:, :lhc.Y_dim] = lhc.c
    c_gamma[:, lhc.Y_dim:] = lhc.gamma
    tmp = mat_exp - lhc.id_mat
    a_row = lhc.a.ravel()          
    return -(a_row @ c_gamma @ (lhc.A_star_inv @ tmp)).ravel()

@njit
def psi_D_star(lhc, t, t_M):
    dt = t_M - t
    mat_exp = mat_exp_approx(lhc.A_star, dt)
    c_gamma = np.zeros((lhc.Y_dim, lhc.Y_dim + lhc.m))
    c_gamma[:, :lhc.Y_dim] = lhc.c
    c_gamma[:, lhc.Y_dim:] = lhc.gamma
    term1 = dt * (lhc.A_star_inv @ mat_exp)
    term2 = lhc.A_star_inv @ ((lhc.id_mat * t - lhc.A_star_inv) @ (mat_exp - lhc.id_mat))
    a_row = lhc.a.ravel()
    return -(a_row @ c_gamma @ (term1 + term2)).ravel()

@njit
def psi_prot(lhc, t, t0, t_M):
    return (1.0 - lhc.delta) * (psi_D(lhc, t, t_M) - psi_D(lhc, t, t0))

@njit
def psi_prem(lhc, t, t0, t_M):
    sum_Z = np.zeros(lhc.Y_dim + lhc.m)
    sum_D = np.zeros(lhc.Y_dim + lhc.m)
    t_grid_len = int(np.round((t_M - t) / lhc.tenor).item()) + 1
    t_grid = np.zeros(t_grid_len)
    for i in range(t_grid_len):
        t_grid[i] = t0 + i * lhc.tenor
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
    result = np.zeros((t_mat_grid.shape[0],lhc.m), dtype=np.float64)
    for i in range(t_mat_grid.shape[0]):
        # Pass a scalar from the array X
        prem = psi_prem(lhc,t,t0,t_mat_grid[i])
        prot = psi_prot(lhc,t,t0,t_mat_grid[i])
        for z_dim in range(lhc.Y_dim,lhc.m+lhc.Y_dim):
            term1 = prot[z_dim] / np.dot(prem, chi)
            term2 = np.dot(prot, chi) / np.dot(prem, chi)**2 *prem[z_dim] 
            result[i,z_dim-1] = term1 - term2
        
    return result


@njit
def cds_fun_lin(lhc, chi_m1, t,t0, t_mat_grid):
    chi_m1 = np.append([1],chi_m1)
    result = np.zeros((t_mat_grid.shape[0],int(lhc.Y_dim+ lhc.m)), dtype=np.float64)
    for i in range(t_mat_grid.shape[0]):
        # Pass a scalar from the array X
        prem = psi_prem(lhc,t,t0,t_mat_grid[i])
        prot = psi_prot(lhc,t,t0,t_mat_grid[i])
        result[i,:] = prot / np.dot(prem, chi_m1)
        
    return result

## Try to run on value of CDS instead.
@njit
def cds_value(lhc, t,t0, t_mat_grid,CDS_grid):
    result = np.zeros((t_mat_grid.shape[0],int(lhc.Y_dim+ lhc.m)), dtype=np.float64)
    for i in range(t_mat_grid.shape[0]):
        # Pass a scalar from the array X
        prem = psi_prem(lhc,t,t0,t_mat_grid[i])
        prot = psi_prot(lhc,t,t0,t_mat_grid[i])
        result[i,:] = prot - CDS_grid[i] * prem
        
    return result


# standard kalman with previous in denom.
@njit
def update_step_cds(X_pred, P_pred, h,h_p, R_k, t_obs,t0, t_mats, lhc, CDS_k):
    pred_Xn = np.append([1],X_pred) # Add one for computations of cds spread and derivative
    # Step 3: Mean prediction, covariance, Kalman Gain etc.
    mu_k = h(lhc, pred_Xn, t_obs,t0, t_mats)
    
    H_x = h_p(lhc, pred_Xn, t_obs,t0, t_mats)
    # covariance
    S_k = H_x @ P_pred @ H_x.T + R_k
    det_S = np.linalg.det(S_k)
    if abs(det_S) < 1e-12:
        S_k_inv = np.linalg.pinv(S_k)
    else:
        S_k_inv = np.linalg.inv(S_k)

    # Step 4: Compute Kalman Gain, filtered mean state, covariance.
    K_k = P_pred @ H_x.T @ S_k_inv
    vn = (CDS_k - mu_k) # In Linear Approx instance
    m_k = X_pred + K_k @ vn
    P_k = P_pred - K_k @ S_k @ K_k.T

    return mu_k, vn,S_k, m_k, P_k

def update_step_lin(X_pred, P_pred, h, R_k, t_obs,t0, t_mats, lhc, CDS_k):
    L = X_pred.shape[0]
    # Step 3: Mean prediction, covariance, Kalman Gain etc.
    # H = h(lhc, X_pred_m1, t_obs,t0, t_mats)
    H = h(lhc, t_obs,t0, t_mats, CDS_k) # Using (14) instead
    mu_k = H[:,1:] @ X_pred + H[:,0]

    # covariance
    S_k = H[:,1:] @ P_pred @ H[:,1:].T + R_k
    det_S = np.linalg.det(S_k)
    if abs(det_S) < 1e-12:
        S_k_inv = np.linalg.pinv(S_k)
    else:
        S_k_inv = np.linalg.inv(S_k)

    # Step 4: Compute Kalman Gain, filtered mean state, covariance.
    K_k = P_pred @ H[:,1:].T @ S_k_inv
    # vn = (CDS_k - mu_k) # In Linear Approx instance
    vn = (np.zeros(CDS_k.shape) - mu_k) # In CDS Value instanse
    m_k = X_pred + K_k @ vn
    P_k = P_pred - K_k @ S_k @ K_k.T

    return mu_k, vn,S_k, m_k, P_k

# Prediction step:
@njit
def prediction_step_lin(Xn, Pn, Matrix,const, Q_k):
    # Step 3: Mean prediction and cov predition.
    m_k = Matrix @ Xn + const 

    # covariance - 
    # Should be zero at first, then fill non Y cols. Same as done in Q_k
    P_k = np.zeros(shape = Q_k.shape)
    # P_k[1:,1:] = h[1:,1:] @ Pn[1:,1:] @ h.T[1:,1:] + Q_k[1:,1:]
    P_k = Matrix @ Pn @ Matrix.T + Q_k

    return m_k, P_k


# Conditional variance as if we had gaussian noise (constant Sigma).
@njit
def get_cond_var(A, Sigma_prod, Delta):
    # Is logic can work completeley, just consider  A'=-A. Then same form
    A_mod = - A
    Lambda, E = np.linalg.eig(A_mod) # Just eigenvalues as array, columns are eigenvectors.
    
    S_bar = np.linalg.inv(E) @ Sigma_prod @ np.linalg.inv(E).T

    dim = A_mod.shape[0] # assume every matrix is of same size and shape.

    # Then V_Delta. 
    V_Delta = np.zeros((dim,dim))
    V_Delta_inf = np.zeros((dim,dim))

    for i in range(0,dim):
        for j in range(0,dim):
            exp_factor = (1-np.exp(-(Lambda[i] + Lambda[j]) * Delta))
            V_Delta[i,j] = S_bar[i,j] * exp_factor / (Lambda[i] + Lambda[j])
            V_Delta_inf[i,j] = S_bar[i,j] / (Lambda[i] + Lambda[j])

    Var = E @ V_Delta @ E.T
    Var_inf = E @ V_Delta_inf @ E.T


    return Var, Var_inf

@njit
def build_P_params(params,gamma,lhc_p):
        gamma1 = gamma # due to parametrization
        kappa, theta = params[:lhc_p.m],params[lhc_p.m:2*lhc_p.m]
        sigma_i,sigma_err = params[2*lhc_p.m:3*lhc_p.m], params[-1]
        r = lhc_p.r
        Y_dim = lhc_p.Y_dim
        delta = lhc_p.delta
        tenor = lhc_p.tenor 
        # Rebuild P parameters.
        lhc = rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor)


        return lhc,kappa,theta, sigma_i, sigma_err



@njit
def build_matrices(lhc,sigma_i,sigma_err,n_mat):
    A_trans = np.zeros((lhc.Y_dim + lhc.m, lhc.Y_dim + lhc.m))
    A_trans[:lhc.Y_dim, :lhc.Y_dim] =  lhc.c # np.ones(shape = lhc.c.shape) 
    A_trans[:lhc.Y_dim, lhc.Y_dim:] = lhc.gamma
    A_trans[lhc.Y_dim:, :lhc.Y_dim] = lhc.b
    A_trans[lhc.Y_dim:, lhc.Y_dim:] =  lhc.beta 

    # Get covariance. 
    sigma = np.zeros((lhc.Y_dim + lhc.m,lhc.m))
    sigma[1:,0:] = np.diag(sigma_i)

    cov_trans = sigma 

    cov_meas =  np.identity(n = int(n_mat)) * sigma_err**2


    return A_trans,cov_trans,cov_meas



### Actual kalman filters.


# @njit
# def drift_term(Xn_prev,lhc_p,Delta):
#     Matrix = np.ones(Xn_prev.shape[0]) + ((lhc_p.beta + np.identity(lhc_p.beta.shape[0])*(-lhc_p.gamma @ Xn_prev)))*Delta
#     const = lhc_p.b.flatten() * Delta
#     return Matrix, const

@njit
def drift_term(Xn,lhc_p,Delta):
    Matrix = Xn + ((lhc_p.beta + np.identity(lhc_p.beta.shape[0])*(-lhc_p.gamma @ Xn))@ Xn)*Delta
    const = lhc_p.b.flatten() * Delta
    return Matrix + const

# Drift derivative.
# @njit
# def drift_deriv_term(Xn,lhc_p,Delta):
#     # Matrix = (np.ones(lhc_p.beta.shape[0]) + 
#     #           ((lhc_p.beta @ np.ones(lhc_p.beta.shape[1]) + 
#     #             (-lhc_p.gamma @ Xn) * np.identity(lhc_p.beta.shape[0]) @ np.ones(lhc_p.beta.shape[0])+ 
#     #             (-lhc_p.gamma) * np.identity(lhc_p.beta.shape[0]) @ Xn))*Delta)
#     Matrix = (np.identity(lhc_p.beta.shape[0]) + 
#               (lhc_p.beta @ np.ones(lhc_p.beta.shape[1]) + 
#                 (-lhc_p.gamma @ Xn*2) * np.identity(lhc_p.beta.shape[0]) @ np.ones(lhc_p.beta.shape[0]))
#                 *Delta)
#     return Matrix
@njit
def drift_deriv_term(Xn, lhc_p, Delta):
    x = Xn.reshape(-1)
    B = lhc_p.beta
    g = lhc_p.gamma.flatten()   # ensure 1D
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
    # RETHINK THIS A LOT. SEEMS LIKELY THAT THERE IS SOME SORT OF ERROR HERE. 
    n_obs = len(t_obs)
    n_mat = T_M_grid.shape[0]

    # Define initial values
    X0 = np.ones(shape=(lhc.m,)) *X0
    X = np.ones((lhc.m, n_obs))
    Y = np.ones((n_obs)) # Implicitly sets Y0
    # Time 0 values
    X[:, 0] = X0
    # Previous Z, starting guess
    Z = np.ones((lhc.m,n_obs))
    Y_prev = Y[0]
    X_prev = X[:,0]
    Z_prev = X[:,0] / Y[0]


    ti = t_obs[1]
    ti_prev = t_obs[0]
    dt = ti-ti_prev
    # Find Z,X,Y
    for time_idx in range(0, n_obs):
        A_big = np.empty(shape = (n_mat,lhc.m))
        y_big = np.empty(shape = (n_mat))
        # Weight matrix
        W = np.zeros(shape=(n_mat,n_mat))
        ti = t_obs[time_idx]
        # Build stacked vector
        one_Z = np.empty(shape = (1 + lhc.m))
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
        if (lhc.m == 1) & (n_mat == 1):
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
            Y[time_idx] = Y_prev + dt * (lhc.gamma.flatten() @ X_prev)
        
        X[:,time_idx] = Y[time_idx] * Z[:,time_idx]


        # Bump previous value
        Z_prev = Z[:,time_idx]
        X_prev = X[:,time_idx]
        Y_prev = Y[time_idx]


    return X,Y,Z


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
        self.gamma1 = rng.uniform(0.2, 0.5, size=(Y_dim,))       # gamma1 strictly pos.
        self.kappa = rng.uniform(self.gamma1, 0.9, size=(X_dim,))       # Kappa given 
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
        t_grid_len = int(np.floor((t_M - t) / self.tenor).item()) + 1
        t_grid = np.zeros(t_grid_len)
        for i in range(t_grid_len):
            t_grid[i] = t + i * self.tenor
        for j in range(1, t_grid_len):
            dt = t_grid[j] - t_grid[j-1]
            sum_Z += dt * self.psi_Z(t, t_grid[j])
            if j < t_grid_len - 1:
                sum_D += dt * self.psi_D(t, t_grid[j])
        return (sum_Z + self.psi_D_star( t, t_M) - self.psi_D_star(t, t0)
                + t_grid[-2] * self.psi_D(t, t_M) - sum_D - t0 * self.psi_D( t, t0))

    def psi_cds(self, t, t0, t_M, k):
        return self.psi_prot(t, t0, t_M) - k * self.psi_prem(t, t0, t_M)


    def CDS_model(self,t_obs, T_M_grid, CDS_obs, t0=None, X=None,Y=None,Z=None):
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
        if (X is None) | (Y is None) :
            X,Y,Z = get_states(lhc, t_obs, T_M_grid, CDS_obs,self.X0)

        if Z is not None:
            X,Y = self.kalman_X_Y(t_obs,Z)

        #print('Done Getting Z,Y,X')
        state_vec = np.vstack([Y, X])

        # Here formula is the t_obs according to formula
        CDS = get_CDS_Model(t_obs, t_obs, T_M_grid, state_vec, lhc )
        #print('Done Getting CDS Rate')
        return CDS.T  # NOTE: CHANGED SIGN HERE. No idea why necessary.

    def get_states(self,t_obs, T_M_grid, CDS_obs):
        # Get latent states.
        t0 = t_obs
        mat_actual = np.array([[0.2137 + i, 0.4658 + i,0.7178 + i,0.9671+i ] 
                                for i in range(0,int(np.max(t0)+1))]).flatten()
        # Ensure mat_actual is sorted
        mat_actual_sorted = np.sort(mat_actual)

        # For each element in t_mat_grid, find the smallest mat_actual that is >= element
        t0 = np.array([mat_actual_sorted[np.searchsorted(mat_actual_sorted, val, side='left')] 
                                        for val in t0.flatten()]).reshape(t0.shape)

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

        # result = minimize(
        #     fun=self.objective,
        #     x0=flat_init,
        #     args=(t_obs, T_M_grid, CDS_obs),
        #     method='SLSQP',
        #     constraints=constraints,
        #     tol = 1e-05
        # )

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
    def build_P_params(self,params=None, gamma1=None,rng=None):
        # NOTE: Gamma does not change!
        if params is None:
            if rng is None:
                rng = np.random.default_rng()  # independent each time
            Y_dim, X_dim = self.Y_dim, self.m
            gamma1 = self.gamma1[0]

            # Set inital values. Need to comply with (38)
            kappa = rng.uniform(gamma1, 0.99, size=(X_dim,))       # Kappa given 
            theta = np.zeros(X_dim)

            for i in range(0,self.m):
                theta[i] = rng.uniform(1e-6, 1-gamma1/kappa[i], size=(1,))       # Theta coeffs
            
            ### New stuff: All the ones needed here e.g. sigma, sigma_Err
            sigma_i = rng.uniform(0.1, 0.15, size=(X_dim,))       # Kappa given 
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
            kappa, theta = params[:self.m],params[self.m:2*self.m]
            sigma_i,sigma_err = params[2*self.m:3*self.m], np.array([params[-1]])
            r = self.r
            Y_dim = self.Y_dim
            delta = self.delta
            tenor = self.tenor 
            # Rebuild P parameters.
            lhc = rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor)

        self.kappa_p,self.theta_p, self.sigma, self.sigma_err=kappa,theta, sigma_i, sigma_err

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

    def kalmanfilter_opt(self,params, t_obs,t0,T_M_grid,CDS_obs,lhc_p,X0):
        print(params)
        # Define the parameters already to be able to look over them
        gamma1 = params[2*lhc_p.m]
        params_p = params[2*lhc_p.m+1:]
        lhc_p,kappa_p,theta_p, sigma, sigma_err = build_P_params(params_p,gamma1,lhc_p)
        params_q = params[:2*lhc_p.m+1]
        kappa, theta, gamma1 = params_q[:lhc_p.m],params_q[lhc_p.m:2*lhc_p.m], params_q[-1]

        # Get initial guesses.
        n_obs = CDS_obs.shape[0]
        n_mat = CDS_obs.shape[1]

        # Build Q params.
        r = lhc_p.r
        Y_dim = lhc_p.Y_dim
        delta = lhc_p.delta
        tenor = lhc_p.tenor 
        lhc_q = rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor)

        # Rebuild p again as needed later on.
        lhc_p = rebuild_lhc_struct(kappa_p, theta_p, gamma1, r, Y_dim, delta, tenor)

        Delta = t_obs[1] - t_obs[0] # Only apprx for now. Move to loop maybe.

        # Only A_trans utilzes new params
        A_trans,Sigma,R_k = build_matrices(lhc_p,sigma,sigma_err,n_mat)

        L = int(lhc_p.m)
        log_likelihood = 0

        # Just to set Xn,Pn, but not needed to be thse vals.
        # Don't know these values. Just arbitrary guessing. on X. Remainder calc.
        # NOTE: SHOULD BE INITIALIZED AT EXPECTED VAL.
        Y0 = np.array([1])
        X0 = np.ones(shape=(lhc_q.m,)) * X0
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
        mu1 = self.solve_mu1(kappa_p, theta_p, gamma1)
        mu = self.compute_stationary(kappa_p,theta_p,lhc_q.m,gamma1,mu1= mu1)
        pred_Xn = mu #drift_term(mu,lhc_p,Delta) #mu
        # Z0[1:] + (lhc_p.b.flatten() + (lhc_p.beta + np.identity(lhc_p.beta.shape[0]) * (-lhc_p.gamma @ Z0[1:])) @ Z0[1:])*Delta
        # Just set the covariance guess to the innovated one.
        P_state = np.array([mu[i] * (1 - mu[i]) for i in range(0,mu.shape[0]) ])
        # Initial Cov Prediction.Z0
        # Sigma = Sigma[1:,:]
        Sigma_prod = (Sigma @ np.diag(np.sqrt(P_state))) @ (Sigma @ np.diag(np.sqrt(P_state))).T 
        # Just an attemt, not to keep.
        P0 = Sigma_prod.copy()
        pred_Pn = P0[1:,1:] 
        
        # Run algo. 
        for n in range(0,n_obs):
            # if n == 0:
            #     pred_Xm1 = mu
            Zn[n,:], vn,S_k, Xn[n,:], Pn[n,:,:] = update_step_cds(pred_Xn,pred_Pn,cds_fun,cds_deriv,R_k,
                                                             t_obs[n],t0[n],T_M_grid[:,n],lhc_q,CDS_obs[n,:])
            # Zn[n,:], vn,S_k, Xn[n,:], Pn[n,:,:] = update_step_lin(pred_Xn,pred_Pn,cds_value,R_k,
            #                                                     t_obs[n],t0[n],T_M_grid[:,n],lhc_q,CDS_obs[n,:])
            Xn_extended = np.append([1],Xn[n,:])
            Zn[n,:] =  cds_fun(lhc_q,Xn_extended,t_obs[n],t0[n],T_M_grid[:,n])
            # pred_Xm1 = pred_Xn.copy()

            # add additional check until parameters are figured out
            if (np.any(Xn<0) |np.any(Xn>1)):
                return 1e12,Xn, Zn, Pn 

            # Update log likelihood.            
            det_S = np.abs(np.linalg.det(S_k))
            try: 
                S_inv = np.linalg.inv(S_k) 
            except:
                S_inv = np.linalg.pinv(S_k)

            log_likelihood += - 0.5 * (S_k.shape[0] * np.log(2*np.pi) + np.log(det_S) +
                                        vn.T @ S_inv @ vn
            )

            if (n < n_obs - 1): # Not sensible to predict further.
                Delta = t_obs[n+1] - t_obs[n] # Only apprx for now. Move to loop maybe.
                # Qt needs modification in according to being stat edependent.
                P_state = np.array([Xn[n,i] * (1 - Xn[n,i]) for i in range(0,Xn.shape[1]) ])
                Sigma_prod = (Sigma @ np.diag(np.sqrt(P_state))) @ (Sigma @ np.diag(np.sqrt(P_state))).T 
                Q_k = Sigma_prod[1:,1:].copy() * Delta
                
                # Then update the predictions:
                # linear version
                # Do prediction directly, no function
                Xn_prev = Xn[n-1,:]
                if n == 0:
                    Xn_prev = pred_Xn # only sensible guess in this case
                # Taylor approximate the transition. 
                # pred_Xn = drift_term(Xn_prev,lhc_p,Delta)+(
                #     drift_deriv_term(Xn_prev,lhc_p,Delta)@ (Xn[n,:]-Xn_prev))
                # Pure Euler
                pred_Xn = drift_term(Xn[n,:],lhc_p,Delta) 

                P_cov = drift_deriv_term(Xn[n,:],lhc_p,np.sqrt(Delta))
                pred_Pn =  P_cov @ Pn[n,:,:] @ P_cov.T + Q_k

                # Straight euler.
                # Matrix,const = drift_term(Xn_prev,lhc_p,Delta)
                # pred_Xn, pred_Pn = prediction_step_lin(Xn[n,:],Pn[n,:,:],Matrix,const,Q_k)

        return - log_likelihood, Xn, Zn, Pn 

    def kalman_wrapper(self,params, t_obs,t0,T_M_grid,CDS_obs,lhc_p,X0):
            # Define the parameters already to be able to look over them
        gamma1 = params[2*lhc_p.m]
        params_p = params[2*lhc_p.m+1:]
        lhc_p,kappa_p,theta_p, sigma, sigma_err = build_P_params(params_p,gamma1,lhc_p)
        params_q = params[:2*lhc_p.m+1]
        kappa, theta, gamma1 = params_q[:lhc_p.m],params_q[lhc_p.m:2*lhc_p.m], params_q[-1]

            # --------- HARD CONSTRAINT CHECKS ---------
        # 1. Positivity - must hold for all params. We use hard constraint from paper.
        if np.any(params <= 0):
            return 1e12  # infeasible, huge loss

        # 2. Custom constraints

        g1 = theta[-1] * kappa[-1]  - sigma[-1]**2/2 
        if g1 < 0:
            return 1e12

        g2 = theta_p[-1] * kappa_p[-1]  - sigma[-1]**2/2 
        if g2 < 0:
            return 1e12


        for i in range(lhc_p.m):
            g3 = gamma1 - kappa[i] + kappa[i] * theta[i] + sigma[i]**2/2
            if g3 > 0:   
                return 1e12

        for i in range(lhc_p.m):
            g4 = gamma1 - kappa_p[i] + kappa_p[i] * theta_p[i]   + sigma[i]**2/2
            if g4 > 0:
                return 1e12

        neg_loglik,Xn, Zn, Pn = self.kalmanfilter_opt(params, t_obs,t0,T_M_grid,CDS_obs,lhc_p,X0)
        return neg_loglik

    def get_kalman_params(self,t_obs, T_M_grid, CDS_obs,x0,lhc_p):
        t0 = t_obs
        # mat_actual = np.array([[0.2137 + i, 0.4658 + i,0.7178 + i,0.9671+i ] 
        #                         for i in range(0,int(np.max(t0)+1))]).flatten()
        # # Ensure mat_actual is sorted
        # mat_actual_sorted = np.sort(mat_actual)

        # # For each element in t_mat_grid, find the smallest mat_actual that is >= element
        # t0 = np.array([mat_actual_sorted[np.searchsorted(mat_actual_sorted, val, side='left')] 
        #                                 for val in t0.flatten()]).reshape(t0.shape)


        result = minimize(
            fun=self.kalman_wrapper,
            x0=x0,
            args=(t_obs, t0, T_M_grid, CDS_obs,lhc_p,self.X0),
            method='Nelder-Mead',
            # method = 'L-BFGS-B', # Finite difference method.
            options = {
                "xatol": 1e-4,
                "fatol": 1e-4,
                "maxiter": 500,
                "disp": True
            }
        )

        # Then ready to optimize
        optim_params = result.x
        self.kalman_obj = result.fun
        if self.kalman_obj >=1e12:
            return optim_params, 0, 0, 0
        else:
            neg_log_lik, Xn,Zn, Pn = self.kalmanfilter_opt(optim_params,t_obs,t0,T_M_grid,
                                                                    CDS_obs,lhc_p,self.X0)
            return optim_params, Xn,Zn, Pn

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
            lhc_p = self.build_P_params(params=None, gamma1=None,rng=rng)
            # Set the error to be the Stddeviation of CDS_obs
            #self.sigma_err = np.std(CDS_obs).flatten()
            # Flatten for scipy. 
            x0_Q = self.flatten_params()

            x0_P = np.concatenate([
                self.kappa_p.flatten(),
                self.theta_p.flatten(),
                self.sigma.flatten(),
                self.sigma_err
            ])

            x0 = np.concatenate([x0_Q,x0_P])

            # Test several random points. 
            optim_params,  Xn,Zn, Pn= self.get_kalman_params(t_obs,T_M_grid, CDS_obs,x0,lhc_p)
            # Test new constraints

            if (self.kalman_obj < current_objective):
                print(f"New optimal parameters at iteration {i+1}.")
                current_objective = self.kalman_obj
                out_params, Xn_out,Zn_out, Pn_out = optim_params, Xn,Zn, Pn
                self.unflatten_params(out_params[:x0_Q.shape[0]])

        # Set new optimal parameters. 
        return  out_params,  Xn_out,Zn_out, Pn_out

    # Transform Kalman Z parameters:
    def kalman_X_Y(self,t_obs,Z):
        n_obs = t_obs.shape[0]
        # find Z0 (use stationary given params)
        mu1 = self.solve_mu1(self.kappa_p,self.theta_p,self.gamma1)
        Z0 = self.compute_stationary(self.kappa_p,self.theta_p,self.m,self.gamma1,mu1)
        Y = np.ones(n_obs)
        X = np.ones((n_obs, self.m))*Z0
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
    


########### OPTION APPROXIMATION FORMULAS on credit default swaps. ###################

    def get_bBounds(self, t0, t_M,k): 
        b_min = np.sum(np.minimum(self.psi_cds(t0,t0,t_M,k),np.zeros(self.m+1)))
        b_max = np.sum(np.maximum(self.psi_cds(t0,t0,t_M,k),np.zeros(self.m+1)))    

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
    

    # Assume it is the inital price i need in Legendre poly.
    def PriceCDS(self,z, n,t,t0,t_M,k,Y):
        b_min,b_max = self.get_bBounds(t0, t_M,k)  

        pi = 0
        # Loop from 0 to n+1 (n)
        for j in range(n+1):
            f_n = self.f_n(j,t,t0,b_min,b_max,Y)
            GLPoly = GenLegendrePoly(z,j,b_min,b_max)
            pi += f_n * GLPoly
        
        return pi
    

    ###### Montecarlo simulation of processes #############3
    # Below function to simulate the dicretized processes. 
    def simul_latent_states(self, chi0, T,M,n_mat,seed=None):
        delta = T / M
        T_return = np.array([0.00001] + [delta*k for k in range(1,M+1)])
        path_Q = np.ones((M + 1, self.m+self.Y_dim))

        # Set initial value. 
        path_Q[0,:] = chi0
        W_Q = norm.rvs(size = (M,self.m),random_state=seed) # simulate at beginning - faster!

        # Get A matrix. Add argument if timul under P or Q.
        params_Q = np.concatenate([self.kappa,self.theta,self.gamma1])
        # If params have been sat with meaningfull values
        params_P = np.concatenate([self.kappa_p,self.theta_p, self.sigma, self.sigma_err])
        # If explicitly given, manual sigma is used.
        params = np.concatenate([params_Q,params_P])
        # Set also P params. 
        lhc_p = self.build_P_params(params[2*self.m+1:],self.gamma1)
        _,cov_trans,_ = build_matrices(lhc_p,self.sigma,self.sigma_err,n_mat)
    
        self.rebuild_dynamics()
        # Get A
        A = self.A

        for i in range(1,M+1):
            mu_t = A @ path_Q[i - 1,:]
            # Create Sigma:
            P_state = np.array(path_Q[i - 1,1:] * (path_Q[i-1,0] - path_Q[i-1,1:]))
            Sigma_prod = (cov_trans @ np.diag(np.sqrt(P_state))) 
            # Out
            path_Q[i,:] = path_Q[i-1,:] + delta*mu_t +  np.sqrt(delta) * Sigma_prod @ W_Q[i-1,:]

        return T_return, path_Q
    

    # Simulate option prices in the model. 
    def get_cdso_pric_MC(self,t,t0,t_M,strikes,chi0,N,M,seed=1000, P_params=None):
        # If p params specific (Sigma specific), se can calculate for differnt sigma
        if P_params is not None:
            lhc_p = self.build_P_params(P_params,self.gamma1)

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
                    Value_CDS = self.psi_cds(t0, t0, t_M, strikes[j]) @ latent_end
                    # Discount back. Also, divide by a^TY_t. If t=0, then not matter.
                    # Note still an option, so only enter if positive. 
                    prices[i,j] = np.exp(-self.r * (t0 - t)) * np.maximum(Value_CDS,0) / S[0]
            # Achieve a running mean also for convergence assessment.
            prices_MC_hist[i, :] = np.mean(prices[:i+1, :], axis=0)
            seed += 1

        price_MC = np.mean(prices,axis=0)

        return prices_MC_hist,price_MC
    

    # Simulate digital Barrier in the model. 
    def get_digital_barrier_price_MC(self,t,t0,t_M,T,barrier,chi0,N,M,seed=1000, P_params=None):
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
        prices = np.zeros(shape = N)
        for i in range(N):
            # Get Latent states. Simulate path of CDS till mat.
            T_return, X_Q = self.simul_latent_states( chi0,T,M,n_mat=1,seed=seed)
            # Retrieve price path...
            CDS_sim = self.CDS_model(T_return, T_M_grid=(T_return + t_M).T,CDS_obs=None,t0=t0,Z=X_Q)
            S = X_Q[:,0]
            # Determine if default or not at t0. If S_t \leq unif(1) option payoff is zero.
            U = uniform.rvs()
            # If survival falls below U at any points, default happened prior to T. 
            # Should be zero, but depends on barrier. Default happens at some point below..
            default_event =  S <= U
            if np.any(default_event):
                # In this instance, default has happened as some point. Find index. 
                idx = np.where(default_event)
                # Get path maximum up to the point:
                max_cds_to_default = np.max(CDS_sim[:idx])
                if max_cds_to_default >= barrier:
                    # In this case, there is a payoff of 1, discount back from expiry(pay date) to today
                    prices[i] = np.exp(-self.r * (T - t))
                else:
                    # If not above, there is zero payoff.
                    prices[i] = 0
            # Else - no default happened, but same logic as before.
            else: 
                # Get path maximum up to expiry:
                max_cds_to_default = np.max(CDS_sim)
                if max_cds_to_default >= barrier:
                    # In this case, there is a payoff of 1, discount back from expiry(pay date) to today
                    prices[i] = np.exp(-self.r * (T - t))
                else:
                    # If not above, there is zero payoff.
                    prices[i] = 0
            
            seed += 1
        price_MC = np.mean(prices)

        return price_MC
    
    # Simulate digital Barrier in the model. 
    def get_lookback_price_MC(self,t,t0,t_M,T,chi0,N,M,seed=1000, P_params=None):
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
        prices = np.zeros(shape = N)
        for i in range(N):
            # Get Latent states. Simulate path of CDS till mat.
            T_return, X_Q = self.simul_latent_states( chi0,T,M,n_mat=1,seed=seed+i)
            # Retrieve price path...
            CDS_sim = self.CDS_model(T_return, T_M_grid=(T_return + t_M).T,CDS_obs=None,t0=t0,Z=X_Q)
            S = X_Q[:,0]
            # Determine if default or not at t0. If S_t \leq unif(1) option payoff is zero.
            U = uniform.rvs()
            # If survival falls below U at any points, default happened prior to T - No payoff
            default_event =  S <= U
            if np.any(default_event):
                prices[i] = 0
            # Else - no default happened
            else: 
                # Get path minimum of CDS:
                min_cds_to_default = np.min(CDS_sim)
                prices[i] = np.exp(-self.r * (T - t)) * (CDS_sim[-1] - min_cds_to_default)
            
            seed += 1
        price_MC = np.mean(prices)

        return price_MC
    

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
            Le_np1 = (2*n_current + 1) * x * Le_n / (n_current + 1) - n_current * Le_nm1 / (n_current+1)
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

    # TODO: Find out which n to use here in actuality.
    mathL = np.sqrt((1+2*n)/(2*sigma**2)) * LegendrePoly((x - mu) / sigma,n)

    return mathL




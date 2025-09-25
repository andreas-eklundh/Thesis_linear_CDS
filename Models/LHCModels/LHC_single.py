import numpy as np 
from scipy.optimize import minimize, NonlinearConstraint 
from scipy.stats import norm 
from scipy.integrate import solve_ivp 
from scipy.linalg import expm 
from scipy.optimize import lsq_linear 
from numba import njit, float64, int64 
from numba.experimental import jitclass 
from scipy.integrate import quad
import copy
from scipy.linalg import sqrtm


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
        t_grid[i] = t + i * lhc.tenor
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
def cds_fun_lin(lhc, chi_m1, t,t0, t_mat_grid):
    result = np.zeros((t_mat_grid.shape[0],int(lhc.Y_dim+ lhc.m)), dtype=np.float64)
    for i in range(t_mat_grid.shape[0]):
        # Pass a scalar from the array X
        prem = psi_prem(lhc,t,t0,t_mat_grid[i])
        prot = psi_prot(lhc,t,t0,t_mat_grid[i])
        result[i,:] = prot / np.dot(prem, chi_m1)
        
    return result

#UNSCENTED IMPLEMENTATION
@njit
def update_step(X_pred, P_pred, h, R_k, t_obs,t0, t_mats, lhc, CDS_k,
                     alpha=1e-03, kappa_u=0.0, beta=2.0):
    L = X_pred.shape[0]
    kappa_u = 3 - L
    lam = alpha**2 * (L + kappa_u) - L
    c = L + lam

    # sigma points
    s = np.sqrt(c)* matrix_sqrt(P_pred) 
    # 1. Form sigma points
    chis = np.ones((L,2*L+1))
    chis[:,0] = X_pred
    for i in range(1,L+1):
        chis[:,i] = X_pred + s[:,i-1]
        chis[:,L+i] = X_pred - s[:,i-1]

    # weights
    wm = np.zeros(2*L+1)
    wm[0] = lam / (lam+L)
    wc = np.zeros(2*L+1)
    wc[0] = lam / (lam+L) + (1-alpha**2+beta)
    for i in range(1, 2*L+1):
        wm[i] = wc[i] = 1 / (2*(lam+L))

    # Step 2: Measurement predictions
    Zi = np.zeros((2*L+1, t_mats.shape[0]))
    for i in range(2*L+1):
        Zi[i,:] = h(lhc, chis[:,i], t_obs,t0, t_mats)

    # Step 3: Mean prediction, covariance, Kalman Gain etc.
    mu_k = np.zeros_like(Zi[0])
    for i in range(2*L+1):
        mu_k += wm[i] *  Zi[i,:]

    # covariance
    S_k = R_k.copy()
    for i in range(2*L+1):
        diff = Zi[i,:] - mu_k
        S_k += wc[i] * np.outer(diff, diff)

    S_k_inv = np.linalg.inv(S_k)

    C_k = np.zeros(shape = (L,t_mats.shape[0]))
    # Shaping here in paper is not completely obvious. 
    for i in range(2*L+1):
        C_k += wc[i] * np.outer((chis[:,i] - X_pred),(Zi[i,:] - mu_k))


    # Step 4: Compute Kalman Gain, filtered mean state, covariance.
    K_k = C_k @ S_k_inv
    vn = (CDS_k - mu_k)
    m_k = X_pred + K_k @ vn
    P_k = P_pred - K_k @ S_k @ K_k.T

    return mu_k, vn,S_k, m_k, P_k

# Prediction step:
@njit
def prediction_step(Xn, Pn, h, Q_k,
                     alpha=1e-03, kappa_u=0.0, beta=2.0):
    L = Xn.shape[0]
    kappa_u = 3 - L
    lam = alpha**2 * (L + kappa_u) - L
    c = L + lam

    # sigma points
    s = np.sqrt(c)* matrix_sqrt(Pn) 
    # 1. Form sigma points
    chis = np.ones((L,2*L+1))
    chis[:,0] = Xn
    for i in range(1,L+1):
        chis[:,i] = Xn + s[:,i-1]
        chis[:,L+i] = Xn - s[:,i-1]

    # weights
    wm = np.zeros(2*L+1)
    wm[0] = lam / (lam+L)
    wc = np.zeros(2*L+1)
    wc[0] = lam / (lam+L) + (1-alpha**2+beta)
    for i in range(1, 2*L+1):
        wm[i]  = wc[i] = 1 / (2*(lam+L))

    # Step 2: Measurement predictions
    Xi = np.zeros((2*L+1, L))
    for i in range(2*L+1):
        Xi[i,:] = h @ chis[:,i] # h(lhc, chis[:,i], t_obs,t0, t_mats)

    # Step 3: Mean prediction and cov predition.
    m_k = np.zeros_like(Xi[0])
    for i in range(2*L+1):
        m_k += wm[i] *  Xi[i,:]
    
    # covariance
    P_k = Q_k.copy()
    for i in range(2*L+1):
        diff = Xi[i,:] - m_k
        P_k += wc[i] * np.outer(diff, diff)

    return m_k, P_k


# standard kalman with previous in denom.
@njit
def update_step_lin(X_pred,X_pred_m1, P_pred, h, R_k, t_obs,t0, t_mats, lhc, CDS_k):
    L = X_pred.shape[0]
    # Step 3: Mean prediction, covariance, Kalman Gain etc.
    H = h(lhc, X_pred_m1, t_obs,t0, t_mats)
    mu_k = H @ X_pred

    # covariance
    S_k = H @ P_pred @ H.T + R_k
    S_k_inv = np.linalg.inv(S_k)

    # Step 4: Compute Kalman Gain, filtered mean state, covariance.
    K_k = P_pred @ H.T @ S_k_inv
    vn = (CDS_k - mu_k)
    m_k = X_pred + K_k @ vn
    P_k = P_pred - K_k @ S_k @ K_k.T

    return mu_k, vn,S_k, m_k, P_k

# Prediction step:
@njit
def prediction_step_lin(Xn, Pn, h, Q_k,
                     alpha=1e-03, kappa_u=0.0, beta=2.0):
    
    # Step 3: Mean prediction and cov predition.
    m_k = h @ Xn # h(lhc, chis[:,i], t_obs,t0, t_mats)


    # covariance
    P_k = h @ Pn @ h.T + Q_k

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

# One kalman filter for optimizing and one for outputting.
@njit
def kalmanfilter_out(params, t_obs,t0,T_M_grid,CDS_obs,lhc_p):
    # Get initial guesses.
    n_obs = CDS_obs.shape[0]
    n_mat = CDS_obs.shape[1]

    # Build Q params.
    kappa, theta, gamma1 = params[:lhc_p.m],params[lhc_p.m:2*lhc_p.m], params[2*lhc_p.m]
    r = lhc_p.r
    Y_dim = lhc_p.Y_dim
    delta = lhc_p.delta
    tenor = lhc_p.tenor 
    lhc_q = rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor)

    # Buld P params:
    params_p = params[2*lhc_p.m+1:]
    lhc_p,kappa_p,theta_p, sigma, sigma_err = build_P_params(params_p,gamma1,lhc_p)

    # Rebuild lhc q set again as needed in later computations.
    lhc_p = rebuild_lhc_struct(kappa_p, theta_p, gamma1, r, Y_dim, delta, tenor)

    Delta = t_obs[1] - t_obs[0] # Only apprx for now. Move to loop maybe.

    # Only A_trans utilzes new params
    A_trans,Q_k,R_k = build_matrices(lhc_p,sigma,sigma_err,n_mat)

    L = int(lhc_p.Y_dim + lhc_p.m)

    # Just to set Xn,Pn, but not needed to be thse vals.
    # Don't know these values. Just arbitrary guessing. on X. Remainder calc.
    Y0 = np.array([1])
    X0 = np.ones(shape=(lhc_q.m,)) *0.2

    chi_0 = np.empty(Y0.shape[0] + X0.size, dtype=np.float64)
    chi_0[:Y0.shape[0]] = Y0
    chi_0[Y0.shape[0]:] = X0.ravel()
    P_state = np.array([chi_0[i] * (chi_0[0] - chi_0[i]) for i in range(1,chi_0.shape[0]) ])
    Sigma_prod = (Q_k @ np.diag(np.sqrt(P_state))) @ (Q_k @ np.diag(np.sqrt(P_state))).T 
    P0 = Sigma_prod.copy()
    _, P0[1:,1:] = get_cond_var(A_trans[1:,1:], Sigma_prod[1:,1:], Delta)
    # _, P0 = get_cond_var(A_trans, Sigma_prod, Delta)


    Pn = np.zeros((n_obs,P0.shape[0],P0.shape[0]))
    Xn = np.zeros((n_obs,L))
    Zn = np.zeros((n_obs,n_mat))
    Xn[0,:] = chi_0
    # Initial Predictions of means and cov
    pred_Xn =  mat_exp_approx(A_trans,Delta) @ chi_0
    pred_Pn = P0

    for n in range(0,n_obs):
        # UPDATE STEP
        # Zn[n,:], vn,S_k, Xn[n,:], Pn[n,:,:] = update_step(pred_Xn,pred_Pn,cds_fun,R_k,
        #                                                     t_obs[n],t0[n],T_M_grid[:,n],lhc_q,CDS_obs[n,:])
        # Linear Kalman
        if n == 0:
            pred_Xm1 = chi_0.copy() #Xn[0,0] = chi_0[0] # Force to be 1 survival process

        Zn[n,:], vn,S_k, Xn[n,:], Pn[n,:,:] = update_step_lin(pred_Xn,pred_Xm1,pred_Pn,cds_fun_lin,R_k,
                                                            t_obs[n],t0[n],T_M_grid[:,n],lhc_q,CDS_obs[n,:])
        # Use prediction in the actual CDS calc.
        Zn[n,:] = cds_fun(lhc_q,Xn[n,:],t_obs[n],t0[n],T_M_grid[:,n])
        pred_Xm1 = pred_Xn.copy()

        if (n < n_obs - 1): # Not sensible to predict further.
            Delta = t_obs[n+1] - t_obs[n] # Only apprx for now. Move to loop maybe.

            # Create arrays based on obs. 
            A_trans,Q_k,R_k  = build_matrices(lhc_p,sigma,sigma_err,n_mat)

            # Qt needs modification in according to being stat edependent.
            P_state = np.array([Xn[n,i] * (Xn[n,0] - Xn[n,i]) for i in range(1,chi_0.shape[0]) ])

            # Above is SigmaSigma.Tin the articles notation.
            # NOTE: WOULD HAVE TO LOOK INTO THIS
            Sigma_prod = (Q_k @ np.diag(np.sqrt(P_state))) @ (Q_k @ np.diag(np.sqrt(P_state))).T 
            Q_k = Sigma_prod.copy()
            Q_k[1:,1:], _ = get_cond_var(A_trans[1:,1:], Sigma_prod[1:,1:], Delta)
            # Q_k, _ = get_cond_var(A_trans, Sigma_prod, Delta)

            # As we use left endpoints of X and Y, we can compute the similar expressions to Christiensen.

            # Then update the predictions:
            f_map = mat_exp_approx(A_trans,Delta) # @ Xn

            # Get point updates and update Z predictions
            #pred_Xn, pred_Pn = prediction_step(Xn[n,:],Pn[n,:,:],f_map,Q_k)     
            # Lin version:
            pred_Xn, pred_Pn = prediction_step_lin(Xn[n,:],Pn[n,:,:],f_map,Q_k)     

    return Xn,Zn, Pn



# One kalman filter for optimizing and one for outputting.
@njit
def kalmanfilter_opt(params, t_obs,t0,T_M_grid,CDS_obs,lhc_p):
    # print(params)
    # Define the parameters already to be able to look over them
    gamma1 = params[2*lhc_p.m]
    params_p = params[2*lhc_p.m+1:]
    lhc_p,kappa_p,theta_p, sigma, sigma_err = build_P_params(params_p,gamma1,lhc_p)
    params_q = params[:2*lhc_p.m+1]
    kappa, theta, gamma1 = params_q[:lhc_p.m],params_q[lhc_p.m:2*lhc_p.m], params_q[-1]
    # --------- HARD CONSTRAINT CHECKS ---------
    # 1. Positivity - must hold for all params.
    if np.any(params <= 0):
        return 1e12  # infeasible, huge loss

    # 2. Custom constraints
    g1 = theta[-1] * kappa[-1]
    if g1 < 0:
        return 1e12

    # Equivalent of the above.
    g2 = theta_p[-1] * kappa_p[-1] 
    if g2 < 0:
        return 1e12

    for i in range(lhc_p.m):
        g3 = gamma1 - kappa[i] + kappa[i] * theta[i]
        if g3 > 0:   # since you wanted  -(...) >= 0 earlier
            return 1e12

    for i in range(lhc_p.m):
        g4 = gamma1 - kappa_p[i] + kappa_p[i] * theta_p[i]
        if g4 > 0:
            return 1e12

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

    L = int(lhc_p.Y_dim + lhc_p.m)
    log_likelihood = 0

    # Just to set Xn,Pn, but not needed to be thse vals.
    # Don't know these values. Just arbitrary guessing. on X. Remainder calc.
    # NOTE: SHOULD BE INITIALIZED AT EXPECTED VAL.
    Y0 = np.array([1])
    X0 = np.ones(shape=(lhc_q.m,)) *0.2
    chi_0 = np.empty(Y0.shape[0] + X0.size, dtype=np.float64)
    chi_0[:Y0.shape[0]] = Y0
    chi_0[Y0.shape[0]:] = X0.ravel()
    P_state = np.array([chi_0[i] * (chi_0[0] - chi_0[i]) for i in range(1,chi_0.shape[0]) ])
    # Initial Cov Prediction.
    Sigma_prod = (Sigma @ np.diag(np.sqrt(P_state))) @ (Sigma @ np.diag(np.sqrt(P_state))).T 
    # Just an attemt, not to keep.
    P0 = Sigma_prod.copy()
    _, P0[1:,1:] = get_cond_var(A_trans[1:,1:], Sigma_prod[1:,1:], Delta)
    # _, P0 = get_cond_var(A_trans, Sigma_prod, Delta)

    # Store predictions. 
    pred_Xn = np.zeros(L)
    pred_Pn = np.zeros((P0.shape[0],P0.shape[0]))

    # Initial Predictions of means and cov
    pred_Xn = mat_exp_approx(A_trans,Delta) @ chi_0
    pred_Pn = P0 

    # Run algo. 
    for n in range(0,n_obs):
        # UPDATE STEP
        # Zn, vn,S_k, Xn, Pn = update_step(pred_Xn,pred_Pn,cds_fun,R_k,t_obs[n],t0[n],T_M_grid[:,n],lhc_q,CDS_obs[n,:])
        #lin version.
        if n == 0:
            pred_Xm1 = chi_0.copy() #Xn[0,0] = chi_0[0] # Force to be 1 survival process

        Zn, vn,S_k, Xn, Pn = update_step_lin(pred_Xn,pred_Xm1,pred_Pn,cds_fun_lin,R_k,
                                                            t_obs[n],t0[n],T_M_grid[:,n],lhc_q,CDS_obs[n,:])
        pred_Xm1 = pred_Xn.copy()
        if n == 0:
            Xn[0] = chi_0[0] # Force to be 1 survival process

        # add additional check until parameters are figured out
        if (np.any(Xn[0] - Xn<0)|np.any(Xn<0) |np.any(Xn>1)):
            return 1e12

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
            # Create arrays based on obs. 
            #A_trans,Sigma,R_k  = build_matrices(lhc_p,sigma,sigma_err,n_mat)

            # Qt needs modification in according to being stat edependent.
            P_state = np.array([Xn[i] * (Xn[0] - Xn[i]) for i in range(1,Xn.shape[0]) ])

            Sigma_prod = (Sigma @ np.diag(np.sqrt(P_state))) @ (Sigma @ np.diag(np.sqrt(P_state))).T 
            Q_k = Sigma_prod.copy()
            Q_k[1:,1:], _ = get_cond_var(A_trans[1:,1:], Sigma_prod[1:,1:], Delta)
            # Q_k, _ = get_cond_var(A_trans, Sigma_prod, Delta)
            # Then update the predictions:
            f_map = mat_exp_approx(A_trans,Delta) # @ Xn

            # Get point updates and update Z predictions
            # pred_Xn, pred_Pn = prediction_step(Xn,Pn,f_map,Q_k)     
            # linear version
            pred_Xn, pred_Pn = prediction_step_lin(Xn,Pn,f_map,Q_k)
    return - log_likelihood 



@njit
def matrix_sqrt(A):
    # Thanks to https://stackoverflow.com/questions/71232192/efficient-matrix-square-root-of-large-symmetric-positive-semidefinite-matrix-in
    D, V = np.linalg.eig(A)
    Bs = (V * np.sqrt(D)) @ V.T
    return Bs


### NOTE: THIS METHODOLODY WILL NOT WORK, STILL NOT OPTIMIZING AT EACH STEP (SO USE PREVIOUS VAL)
@njit
def get_states(lhc, t_obs, T_M_grid, CDS_obs):
    # RETHINK THIS A LOT. SEEMS LIKELY THAT THERE IS SOME SORT OF ERROR HERE. 
    n_obs = len(t_obs)
    n_mat = T_M_grid.shape[0]

    # Define initial values
    X0 = np.ones(shape=(lhc.m,)) *0.2
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

    
    def initialise_LHC(self, Y_dim, X_dim, rng=None):
        if rng is None:
            rng = np.random.default_rng()  # independent each time

        self.Y_dim, self.m = Y_dim, X_dim
        self.a = np.ones((self.Y_dim,1))                                      # Y dim is 1 for LHC
        # Set inital values. Need to comply with (38)
        self.kappa = rng.uniform(0.2, 0.9, size=(X_dim,))       # Kappa given 
        self.gamma1 = np.array([np.min(self.kappa) - 1e-1])       # gamma1 strictly pos.
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


    def CDS_model(self,t_obs, T_M_grid, CDS_obs):
        # Get latent states.
        t0 = t_obs
        mat_actual = np.array([[0.2137 + i, 0.4658 + i,0.7178 + i,0.9671+i ] 
                                for i in range(0,int(np.max(t0)+1))]).flatten()
        # Ensure mat_actual is sorted
        mat_actual_sorted = np.sort(mat_actual)

        # For each element in t_mat_grid, find the smallest mat_actual that is >= element
        t0 = np.array([mat_actual_sorted[np.searchsorted(mat_actual_sorted, val, side='left')] 
                                        for val in t0.flatten()]).reshape(t0.shape)

        # Actual maturity dates. Say at March 20, Jun

        kappa, theta, gamma1 = self.kappa,self.theta,self.gamma1[0]
        r = self.r
        Y_dim = self.Y_dim
        delta = self.delta
        tenor = self.tenor 
        lhc = rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor)

        # New get states functionality:
        # Numba code to generate matrices to solve for.
        X,Y,Z = get_states(lhc, t_obs, T_M_grid, CDS_obs)

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
        X,Y,Z = get_states(lhc, t_obs, T_M_grid, CDS_obs)

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
        
    def build_constraints(self, m):
        cons = []

        # g1 = x[i] * x[m+i] >= 0
        # for i in range(m):
        #     cons.append({'type': 'ineq',
        #                 'fun': lambda x, i=i: x[i] * x[m+i]})
        cons.append({'type': 'ineq',
                        'fun': lambda x: x[m] * x[2*m]})
        # g2: x[-1] - x[i] + x[i]*x[m+i] <= 0  ->  -g2 >= 0
        for i in range(m):
            cons.append({'type': 'ineq',
                        'fun': lambda x, i=i: -(x[-1] - x[i] + x[i]*x[m+i])})

        # Non-negativity
        for i in range(2*m+1):  # or len(x) if variable dimension
            cons.append({'type': 'ineq',
                        'fun': lambda x, i=i: x[i]})
        return cons



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
            self.initialise_LHC( self.Y_dim, self.m, rng=rng)
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
            sigma_err = rng.uniform(0.00001, 0.0001, size=(Y_dim,))       # Kappa given 

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



    def get_kalman_params(self,t_obs, T_M_grid, CDS_obs,x0,lhc_p):
        t0 = t_obs
        mat_actual = np.array([[0.2137 + i, 0.4658 + i,0.7178 + i,0.9671+i ] 
                                for i in range(0,int(np.max(t0)+1))]).flatten()
        # Ensure mat_actual is sorted
        mat_actual_sorted = np.sort(mat_actual)

        # For each element in t_mat_grid, find the smallest mat_actual that is >= element
        t0 = np.array([mat_actual_sorted[np.searchsorted(mat_actual_sorted, val, side='left')] 
                                        for val in t0.flatten()]).reshape(t0.shape)


        result = minimize(
            fun=kalmanfilter_opt,
            x0=x0,
            args=(t_obs, t0, T_M_grid, CDS_obs,lhc_p),
            method='Nelder-Mead',
            # method = 'L-BFGS-B', # Finite difference method.
            options = {
                "xatol": 1e-4,
                "fatol": 1e-4,
                "maxiter": 2000,
                "disp": True
            }
        )

        # Then ready to optimize
        optim_params = result.x
        self.kalman_obj = result.fun

        Xn,Zn, Pn = kalmanfilter_out(optim_params,t_obs,t0,T_M_grid,
                                                                 CDS_obs,lhc_p)
        return optim_params, Xn,Zn, Pn
    
    def run_n_kalmans(self, t_obs,T_M_grid, CDS_obs, base_seed = 1000,  n_restarts = 20):
        # Define grid of values. 
        current_objective = 1e10 #very high objective.
        out_params = self.flatten_params()
        for i in range(n_restarts):
            print(f"Optimization {i+1}")
            rng = np.random.default_rng(base_seed + i)  # deterministic but different
            # Set Q parameters.
            self.initialise_LHC(self.Y_dim,self.m,rng)
            # Get P Parameters /initialise
            lhc_p = self.build_P_params(params=None, gamma1=None,rng=rng)
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
                out_params, Xn,Zn, Pn = optim_params, Xn,Zn, Pn

        # Set new optimal parameters. 
        return out_params, Xn,Zn, Pn


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
            nom = (lhc_p.b - self.b) * Y[n] + ((lhc_p.beta - self.beta) @ X[:,n])
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
    def simul_latent_states(self, Chi0,T,M,measure,seed=None):
        delta = T / M
        T_return = np.array([0.00001] + [delta*k for k in range(1,M+1)])
        path = np.ones((M + 1, self.m+self.Y_dim))
        # Set initial value. 
        path[0,:] = Chi0
        W = norm.rvs(size = (M,self.m),random_state=seed) # simulate at beginning - faster!
        # Get A matrix. Add argument if timul under P or Q.
        params_Q = np.concatenate([self.kappa,self.theta,self.gamma1])
        params_P = np.concatenate([self.kappa_p,self.theta_p, self.sigma, self.sigma_err])
        params = np.concatenate([params_Q,params_P])
        # Set also P params. 
        lhc = self.build_P_params(params[2*self.m+1:],self.gamma1)
        if measure == 'P':
            A,cov_trans,_ = build_matrices(lhc,self.sigma,self.sigma_err,1)
    
        elif measure == 'Q':
            self.rebuild_dynamics()
            # Get A
            A = self.A
            _,cov_trans,_ = build_matrices(lhc,self.sigma,self.sigma_err,1)

        else:
            raise ValueError(f"Unknown measure: {measure}")
        for i in range(1,M+1):
            mu_t = A @ path[i - 1,:]
            # Create Sigma:
            P_state = np.array(path[i - 1,1:] * (path[i-1,0] - path[i-1,1:]))
            Sigma_prod = (cov_trans @ np.diag(np.sqrt(P_state))) 
            # Out
            path[i,:] = path[i-1,:] + delta*mu_t +  np.sqrt(delta) * Sigma_prod @ W[i-1,:]

        return T_return, path
    



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




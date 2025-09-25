import numpy as np
from scipy.integrate import solve_ivp
from numba import njit, float64, int64 
from scipy.linalg import sqrtm

# Consider just having one class, or all Kalman filters here.

@njit
def update_step(X_pred, P_pred, h, R_k, t_obs, t_mats, cir, CDS_k,
                     alpha=5e-01, kappa_u=0.0, beta=2.0):
    L = X_pred.shape[0]
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
        wm[i]  = wc[i] = 1 / (2*(lam+L))

    # Step 2: Measurement predictions
    Zi = np.zeros((2*L+1, t_mats.shape[0]))
    for i in range(2*L+1):
        Zi[i,:] = h(cir, chis[:,i],t_obs, t_mats)

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

    # In principle, should compute some actual estimate of Zn - need to punish. 

    return mu_k, vn,S_k, m_k, P_k

# Prediction step:
@njit
def prediction_step(Xn, Pn, h, Q_k,phi_0,phi_X,
                     alpha=0.5, kappa_u=0.0, beta=2.0):
    L = Xn.shape[0]
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
        Xi[i,:] = h(phi_0,phi_X,chis[:,i]) # h(lhc, chis[:,i], t_obs,t0, t_mats)

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



# Remember, there is an optimization step, and a get states step. Optimizer for itself. 
@njit
def KalmanUnscented(params, cir_params,t_obs, t_mat_grid, CDS,h):
    kappa, theta, sigma,kappa_p,theta_p,sigma_err = params[0], params[1], params[2], params[3],params[4], params[5]
    Sigma = (np.identity(t_mat_grid.shape[0]) *sigma_err**2)
    alpha = 2 * kappa_p* theta_p / sigma**2
    beta = 2 *kappa_p / sigma**2
    #l CIR values. 
    X0 = np.array([alpha / beta])
    P0 = np.array([[alpha / beta**2]])
    # CIR conditional Mean and Variance based on parameters. 
    # Do this correctly at some point. Will depend on values. 
    # Look op CIR Solution, approximate remainder as Gaussian. 

    n_obs = CDS.shape[0]
    n_mat = CDS.shape[1]
    L = X0.shape[0]

    Xn = np.zeros((n_obs,L))
    Zn = np.zeros((n_obs,n_mat))
    Pn = np.zeros((n_obs,L,L))
    
    # We want to store all predictions. 
    pred_Xn  = X0
    pred_Pn  = P0
    Delta = t_obs[1] - t_obs[0] # Only apprx for now. Move to loop maybe.


    # Run algo. 
    for n in range(0,n_obs):
        # UPDATE STEP
        Zn[n,:], vn,S_k, Xn[n,:], Pn[n,:,:] = update_step(pred_Xn,pred_Pn,h,Sigma,
                                                            t_obs[n],t_mat_grid[:,n],cir_params,CDS[n,:])

        # Zn in is actually h(Xn) as Xn is our best guess. 
        Zn[n,:] = h(cir_params,Xn[n,:],t_obs[n],t_mat_grid[:,n])
        # Create arrays based on obs. 
        phi_0 = (1-np.exp(-kappa_p * Delta)) * theta_p
        phi_X = np.exp(-kappa_p * Delta)
        # Use CIR Variance for this (not going to need it)
        # Then update the predictions:
        if (n < n_obs - 1): # Not sensible to predict further.
            Q_t = (sigma**2 * theta_p * (1-np.exp(-kappa_p * Delta))**2 / (2 * kappa_p) + 
                    Xn[n,:] * sigma**2 * (np.exp(-kappa_p * Delta) - np.exp(-2*kappa_p * Delta))/ kappa_p)
            Q_t = Q_t.reshape((1,1))
            Delta = t_obs[n+1] - t_obs[n]
            pred_Xn, pred_Pn = prediction_step(Xn[n,:],Pn[n,:,:],trans_map,Q_t, phi_0,phi_X)     


    return  Xn,Zn, Pn


@njit
def KalmanUnscentedFit(params, cir_params,t_obs, t_mat_grid, CDS,h):
    kappa, theta, sigma,kappa_p,theta_p,sigma_err = params[0], params[1], params[2], params[3],params[4], params[5]
    Sigma = (np.identity(t_mat_grid.shape[0]) *sigma_err**2)
    alpha = 2 * kappa_p* theta_p / sigma**2
    beta = 2 *kappa_p / sigma**2
    #l CIR values. 
    X0 = np.array([alpha / beta]) # theta is unconditional extation.
    P0 = np.array([[alpha / beta**2]]) # unconditional variance. 
    # CIR conditional Mean and Variance based on parameters. 
    # Do this correctly at some point. Will depend on values. 
    # Look op CIR Solution, approximate remainder as Gaussian. . 

    n_obs = CDS.shape[0]

    L = X0.shape[0]
    log_likelihood = 0

    # Just to set Xn,Pn, but not needed to be thse vals.
    Xn = X0
    Pn = P0
    # Store predictions. 
    pred_Xn = np.zeros(L)
    pred_Pn = np.zeros((P0.shape[0],P0.shape[0]))

    # Initial Predictions of means and cov
    pred_Xn = X0
    pred_Pn = P0

    # We want to store all predictions. 
    Delta = t_obs[1] - t_obs[0] # Only apprx for now. Move to loop maybe.


    # Run algo. 
    for n in range(0,n_obs):
        # UPDATE STEP
        # Update latent state
        Zn, vn,S_k, Xn, Pn = update_step(pred_Xn,pred_Pn,h,Sigma,
                                                            t_obs[n],t_mat_grid[:,n],cir_params,CDS[n,:])
        # If unrealistic values, ignore.
        if np.any((Xn < 0)):
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

        # Create arrays based on obs. 
        phi_0 = (1-np.exp(-kappa_p * Delta)) * theta_p
        phi_X = np.exp(-kappa_p * Delta) # Needs some tuning

        if (n < n_obs - 1): # Not sensible to predict further.
            Delta = t_obs[n+1] - t_obs[n] # Only apprx for now. Move to loop maybe.

            # Get point updates and update Z predictions
            # Use CIR Variance to get predictions.
            Q_t = (sigma**2 * theta_p * (1-np.exp(-kappa_p * Delta))**2 / (2 * kappa_p) + 
                Xn * sigma**2 * (np.exp(-kappa_p * Delta) - np.exp(-2*kappa_p * Delta))/ kappa_p)
            Q_t = Q_t.reshape((1,1))


            pred_Xn, pred_Pn = prediction_step(Xn,Pn,trans_map,Q_t, phi_0,phi_X)     
            if np.any((pred_Xn < 0)):
                return 1e12 
    return - log_likelihood

@njit
def trans_map(phi0, phi1, X):
    return phi0 + phi1 *  X


@njit
def matrix_sqrt(A):
    # Thanks to https://stackoverflow.com/questions/71232192/efficient-matrix-square-root-of-large-symmetric-positive-semidefinite-matrix-in
    D, V = np.linalg.eig(A)
    Bs = (V * np.sqrt(D)) @ V.T
    return Bs


import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, Bounds
from scipy.interpolate import CubicSpline
from numba import njit, float64, int64
from Models.ATSMGeneral.ATSM import ATSM
from Models.BaselineCIR_alternative.CIR_numba import calc_cds,calc_CDS_numba
from Models.BaselineCIR_alternative.Gamma_solver import DeterministicGamma
from scipy.stats import norm, ncx2, gamma, expon,uniform
from scipy.integrate import quad
from scipy.linalg import expm
import time
from scipy.interpolate import interp1d

from numba.experimental import jitclass
from scipy.optimize import differential_evolution


## Class to actually call
class CIRIntensity():
    def __init__(self, r, delta, tenor,X_dim=1,cascading=False):
        self.r = r
        self.delta = delta
        self.tenor = tenor
        self.X_dim = X_dim
        self.cascading = cascading
        # Set parameters randomly at first, but so something is present.
        self.set_params(params = None)

    def set_params(self,params,seed=None):
        X_dim = self.X_dim
        # If no parameters are set, use random.
        # Also for sithetalatio    n purposes
        if params is None:
            if seed == None:
                rng = np.random.default_rng()  # independent each time
            else:
                rng = np.random.default_rng(seed)  # independent each time
            # Default (short/med) factors
            self.kappa = rng.uniform(0.3, 2, size=(X_dim,))
            # Order. Mainly for identification to make sense.
            # self.kappa = np.sort(self.kappa)[::-1]
            self.theta = rng.uniform(0.1, 1, size=(X_dim,))

            # Assume same for theta for id.
            # self.theta = np.sort(self.theta)[::-1]


            # Initiate MPR specification. Use positivity bound for positive.
            self.lambda1 = np.zeros(X_dim)
            for i in range(X_dim):
                # self.lambda1[i] = rng.uniform(-self.kappa[i]*self.theta[i], self.kappa[i], 1)
                self.lambda1[i] = rng.uniform(-0.6, self.kappa[i], 1)
                # self.lambda1[i] = rng.uniform(-1, np.min(self.kappa), 1)
            # Sort
            # self.lambda1 = np.sort(self.lambda1)
            # initialise all positive. Note one can plug in same lambda is simplicity needed
            self.kappa_p,self.theta_p = self.build_P_drift(self.lambda1,np.zeros_like(self.lambda1))
            # Set sigma in feller grid.
            # Sigma to be in minimum of feller conds.
            feller_min = np.minimum(np.sqrt(2*self.kappa*self.theta),np.sqrt(2*self.kappa_p*self.theta_p))
            self.sigma = np.zeros(X_dim)
            # choose sigma randomly
            self.sigma[:-1] = rng.uniform(0.001, 0.2, size=(X_dim-1,))
            self.sigma[-1] = rng.uniform(0.001, feller_min[-1], size=(1,))
            # Again, initialise with relatively low error.
            self.sigma_err = rng.uniform(0.0001, 0.001, size=(1,))


        else:
            # Else, asusming some paramter tuning, set then here.
            self.kappa, self.theta, self.sigma,self.lambda1,self.sigma_err = self.unpack_params(params)
            self.kappa_p,self.theta_p = self.build_P_drift(self.lambda1,np.zeros_like(self.lambda1))


        # Set affine model attribtues. if cascading
        if self.cascading:
        # Build matrices for Afine term structure models.
            self.K0 = np.zeros(self.X_dim)
            self.K0[-1] = self.kappa[-1] *  self.theta[-1]
            # K0 = kappa * theta

            self.K1 = - np.identity(self.X_dim) *  self.kappa
            for i in range(self.X_dim):
                if i + 1 < self.X_dim:
                    self.K1[i, i+1] =  self.kappa[i] * self.theta[i]

            # Non diagonal entries.
            self.H0 = np.zeros((self.X_dim,self.X_dim ))
            self.H1 = np.zeros((self.X_dim,self.X_dim,self.X_dim))
            for i in range(self.X_dim):
                self.H1[i,i,i] =  self.sigma[i]**2

    def build_P_drift(self,lambda1,lambda2, kappa=None, theta=None):
            if (kappa is None) & (theta is None):
                kappa = self.kappa
                theta = self.theta
            # This is the kappas and thetas under the new specification.
            kappa_p = kappa - lambda1
            theta_p = (theta *kappa +lambda2 ) / kappa_p
            return kappa_p, theta_p

    def unpack_params(self,params):
        X_dim = self.X_dim
        kappa, theta, sigma = params[:X_dim],params[X_dim:2*X_dim],params[2*X_dim:3*X_dim]
        lambda1, sigma_err = params[3*X_dim:4*X_dim], np.array([params[-1]])
        return kappa,theta,sigma,lambda1,sigma_err
    #### Solve affine equations.

    # For reference, also the solution in Lando (2004).
    # Potentially more handystable (numba). Default is rho=1 as it will be the one we use (maybe?)



    def cir_solution(self,params,x0,T,rho=1):
        # Local copies of kappa, theta to minimize code. Rename to comply with Lando.
        kappa,theta,sigma1,lambda1,sigma_err = self.unpack_params(params)
        gamma = np.sqrt(kappa**2 + 2*sigma1**2*rho)
        # If x0 is one dimensional (intensity), use Lando forthetalas

        if (self.cascading  == False) & (np.all(np.isreal(x0))):
            if isinstance(T,float):
                T_size = 1
            else:
                T_size = T.shape[0]

            # In this case, rho needs to be a vector (rho is rho 1 in actuality)
            alpha, beta = np.zeros((T_size)),np.zeros((T_size,self.X_dim))
            for i in range(self.X_dim):
                beta_nom = (- 2 * rho * (np.exp(gamma[i]*T)-1) +
                                x0[i] * np.exp(gamma[i] * T) * (gamma[i] - kappa[i]) +
                                x0[i] * (gamma[i] + kappa[i]))

                beta_denom = (2 * gamma[i] +
                                (gamma[i] + kappa[i] - x0[i] * sigma1[i]**2) * (np.exp(gamma[i] * T) - 1))

                beta[:,i] = beta_nom / beta_denom

                alpha_log_nom = (2 * gamma[i] * np.exp((gamma[i] + kappa[i] ) * T / 2))

                alpha_log_denom = (2 * gamma[i] +
                                    (gamma[i] + kappa[i] - x0[i] * sigma1[i]**2)*(np.exp(gamma[i] * T)-1))
                alpha += (2 * kappa[i] * theta[i] *
                            np.log(alpha_log_nom/alpha_log_denom)
                            / sigma1[i]**2)

            # Yields a vector of solutions.
            return alpha,beta


        # If correlation or dgeneral solutions:
        # This we set to the cascading version like the LHC model.
        else:
        # Build matrices for Afine term structure models.
            self.K0 = np.zeros(self.X_dim)
            self.K0[-1] = kappa[-1] * theta[-1]
            # K0 = kappa * theta

            self.K1 = - np.identity(self.X_dim) * kappa
            for i in range(self.X_dim):
                if i + 1 < self.X_dim:
                    self.K1[i, i+1] = kappa[i] * theta[i]

            # Non diagonal entries.
            self.H0 = np.zeros((self.X_dim,self.X_dim ))
            self.H1 = np.zeros((self.X_dim,self.X_dim,self.X_dim))
            for i in range(self.X_dim):
                self.H1[i,i,i] = sigma1[i]**2
            # Just assume first factor is default intensity in cascading. Mimic LHC
            rho1 = np.zeros(self.X_dim)
            rho1[0] = 1

            atsm = ATSM(self.K0,self.K1,self.H0,self.H1,rho0=0,rho1=rho1)
            atsm.solve_ODE_system(x0,0,T)

            return atsm.alpha,atsm.beta.T

    def cir_derivatives(self,params,x,T,rho=1):
        # Can work in
        kappa,theta,sigma1,lambda1,sigma_err = self.unpack_params(params)
        gamma = np.sqrt(kappa**2 + 2*sigma1**2*rho)
        if self.cascading == False:
            if isinstance(T,float):
                T_size = 1
            else:
                T_size = T.shape[0]

            alpha_x, beta_x = np.zeros((T_size)),np.zeros((T_size,self.X_dim))
            for i in range(self.X_dim):
                # In this case, rho needs to be a vector (rho is rho 1 in actuality)
                denom = (2 * gamma[i] + (gamma[i] + kappa[i] - x[i] * sigma1[i]**2)*(np.exp(gamma[i] * T)-1))

                # Beta
                bterm1  = - (2*rho*(np.exp(gamma[i]*T)-1)**2 * sigma1[i]**2) / denom**2
                bterm2 = (np.exp(gamma[i] * T)*(gamma[i]-kappa[i]) + (gamma[i]+kappa[i])) / denom
                # This  is tge second term of derivatives from the 'bottom'
                bterm3 = x[i] * (np.exp(gamma[i] * T)*(gamma[i]-kappa[i]) + (gamma[i]+kappa[i])) *sigma1[i]**2 *(np.exp(gamma[i]*T)-1) / denom**2

                beta_x[:,i] = bterm1 + bterm2 + bterm3

                # Alpha.
                alpha_x += 2 * kappa[i] * theta[i] *(np.exp(gamma[i] * T) - 1) / denom
        # In the other case, we will find derivatives numerically.
        else:
            # Get alpha beta for small changes. not too large for num s
            # h = np.array([1e-6]*self.X_dim)
            # Is this true? Do calcs.
            h = np.zeros(self.X_dim)
            h[0] = 1e-6
            alpha_x_h,beta_x_h = self.cir_solution(params,h,T)
            alpha_x_hm,beta_x_hm = self.cir_solution(params,-h,T)
            # Derivative.
            alpha_x, beta_x = (alpha_x_h-alpha_x_hm)/(2*h[0]),(beta_x_h-beta_x_hm)/(2*h[0])

        return alpha_x, beta_x


    # The Laplace Transform
    def Laplace_Transform(self,params,lambda_t, T,x_overwrite=None):
        if x_overwrite is not None:
            x = x_overwrite
        else:
            x = np.zeros(self.X_dim)
        alpha,beta = self.cir_solution(params,x,T,rho=1)
        # Return value of Laplace Transform - Specific vals of w->ZCB price.
        return np.exp(alpha + beta @ lambda_t).flatten()

    ##### Section on all the pricing stuff.
    # Coupon leg 'easy' should be similar to a ATSM
    def calc_coupon_leg(self,params,t,t0,t_mat, lambda_t):
        I = np.zeros(1)
        t_grid = np.arange(t0, t_mat + 1e-12, self.tenor)
        for t_idx in range(1, len(t_grid)):
            expectation = self.Laplace_Transform(params, lambda_t.T, np.array([t_grid[t_idx] - t]))
            I += (t_grid[t_idx]-t_grid[t_idx-1]) * np.exp(-self.r * (t_grid[t_idx] - t)) * expectation
        return I


    # Accrued leg. Think it is going to follow similar to protection leg.
    # so (46 on 23/59) with the additional increment term.
    # Helper function to get the grid.
    def _get_default_grid(self, u, t_grid):
        if u <= t_grid[0]:
            return 0.0
        if u >= t_grid[-1]:
            return t_grid[-1] - t_grid[-2]  # last interval length
        idx = np.searchsorted(t_grid, u) - 1
        return u - t_grid[idx]

    def calc_accrual_leg(self,params,t,t0,t_mat, lambda_t):
        x = np.zeros(self.X_dim)
        t_grid = np.arange(t0, t_mat + 1e-12, self.tenor)
        integrand = lambda u: (np.exp(-self.r * (np.array([u-t]))) * self._get_default_grid(u,t_grid) *
            (self.cir_derivatives(params,x,np.array([u-t]))[0] +
            self.cir_derivatives(params,x,np.array([u-t]))[1] @ lambda_t.T) *
            self.Laplace_Transform(params,lambda_t.T, np.array([u-t]))
        )

        Al_val = 0.0
        for i in range(len(t_grid) - 1):
            a, b = t_grid[i], t_grid[i + 1]
            Al_val += quad(integrand, a, b, epsabs=1e-8, epsrel=1e-8, limit=100)[0]
        return Al_val

    # Protection leg:
    def calc_protection_leg(self,params,t,t0,t_mat, lambda_t):
        x = np.zeros(self.X_dim)

        integrand = lambda u: ((1- self.delta)*np.exp(-self.r * (np.array([u-t]))) * (
            self.cir_derivatives(params,x,np.array([u-t]))[0] +
            self.cir_derivatives(params,x,np.array([u-t]))[1] @ lambda_t.T)*
            self.Laplace_Transform(params,lambda_t.T, np.array([u-t]))
            )
        # prot_val, _ = quad(integrand,t0,t_mat,epsabs=1e-9, epsrel=1e-9)
        t_grid = np.arange(t0, t_mat + 1e-12, self.tenor)

        prot_val = 0.0
        for i in range(len(t_grid) - 1):
            a, b = t_grid[i], t_grid[i + 1]
            prot_val += quad(integrand, a, b, epsabs=1e-4, epsrel=1e-4, limit=100)[0]
        return prot_val



    def calc_CDS(self,params,t,t_mat, lambda_t,t0=None):
        # If no t0 provided, assume at inception
        if t0 is None:
            t0=t
        prot_val = self.calc_protection_leg(params,t,t0,t_mat, lambda_t)
        I1 = self.calc_coupon_leg(params,t,t0,t_mat, lambda_t)
        I2 = self.calc_accrual_leg(params,t,t0,t_mat, lambda_t)

        return prot_val /(I1 + I2 )



    def cds_spread(self, X,params, t, t_mat_grid,t0=None):
        # If no t0 provided, assume at inception
        if t0 == None:
            t0 = t
        result = np.zeros(t_mat_grid.shape[0], dtype=np.float64)

        for i in range(t_mat_grid.shape[0]):
            # Pass a scalar from the array X
            # Make sure that t grid is of size 1 due to logic.
            t_mat = np.array([t_mat_grid[i]])
            result[i] = self.calc_CDS(params,t, t_mat, X,t0)[0] # A
        return result

    def Update_step(self,X_pred, P_pred, A,a, R_k, Y):
        # Step 3: Mean prediction, covariance, Kalman Gain etc.
        theta_k = A @ X_pred + a

        # covariance
        S_k = A @ P_pred @ A.T + R_k
        try:
            S_k_inv = np.linalg.inv(S_k)
        except:
            # Pseudo inverse if not working.
            S_k_inv = np.linalg.pinv(S_k)

        # Step 4: Compute Kalman Gain, filtered mean state, covariance.
        K_k = P_pred @ A.T @ S_k_inv
        vn = (Y - theta_k)
        m_k = X_pred + K_k @ vn
        P_k = P_pred - K_k @ S_k @ K_k.T

        return theta_k, vn,S_k, m_k, P_k

    # Prediction step:
    def Prediction_step(self,Xn, Pn, C,d, Q_k):
        # CIR transition.
        # Step 3: Mean prediction and cov predition.
        m_k = (C @ Xn +  d).flatten()

        # covariance -
        # Should be zero at first, then fill non Y cols. Same as done in Q_k
        P_k = np.zeros(shape = Q_k.shape)
        P_k = C @ Pn @ C.T + Q_k

        return m_k, P_k


    # Kalman filtering.
    def Kalman(self,params,t_obs, t_mat_grid, Y, result = False):
        print(params)
        kappa,theta,sigma,lambda1,sigma_err = self.unpack_params(params)
        kappa_p,theta_p = self.build_P_drift(lambda1,np.zeros_like(lambda1),kappa,theta)
        # Stop optimization if bad initial params.
        # positivity constraints (soft bounds)
        # if np.any(params <= 0):
        #     return 1e12

        # # Feller condition: 2*kappa_p*theta_p - sigma^2 >= 0
        # feller_val = 2 * kappa_p * theta_p - sigma**2
        # if np.any(feller_val < 0):
        #     return 1e12

        # # Add feller under both - will hold also for matrices
        # feller_val = 2 * kappa * theta - sigma**2
        # if np.any(feller_val < 0):
        #     return 1e12

        n_obs = t_mat_grid.shape[1]
        n_mat = t_mat_grid.shape[0]

        Sigma = (np.identity(n_mat) * sigma_err**2)

        # Long term mean for each of the processes.
        alpha = 2 * kappa_p* theta_p / sigma**2
        beta = 2 *(kappa_p) / sigma**2
        # CIR values.
        X0 = alpha / beta
        P0 = alpha / beta**2

        if self.cascading:
            K1 = - np.identity(self.X_dim) *  kappa_p
            for i in range(self.X_dim):
                if i + 1 < self.X_dim:
                    K1[i, i+1] =  kappa_p[i] * theta_p[i]
            # X0 = np.cumsum(theta_p[::-1])[::-1]
            X0 =  np.cumprod(theta_p[::-1])[::-1] 
            _,P0 = self.get_cond_var(-K1,
                                     sigma**2*np.diag(X0),0)


        # CIR conditional Mean and Variance based on parameters.
        L = X0.shape[0]

        Xn = np.zeros((L))
        Zn = np.zeros((n_mat))
        Pn = np.zeros((L,L))
        log_likelihood = np.zeros(n_obs)
        if result == True:
            Xn_out = np.zeros((n_obs,L))
            Zn_out = np.zeros((n_obs,n_mat))
            Pn_out = np.zeros((n_obs,L,L))
        # We want to store all predictions.
        pred_Xn  = X0
        pred_Pn  = (P0).reshape((self.X_dim,self.X_dim))
        Delta = t_obs[1] - t_obs[0] # Only apprx for now. Move to loop maybe.

        # Run algo.
        x0_zcb = np.zeros(self.X_dim)
        kappa_P_diag = np.identity(self.X_dim) * (kappa_p)

        # To speed up, solve ricatti equations. Will be time homogenous.
        # Solve Ricatti Equations. Might move inside loop later - MUCH FASTER OUT HERE, IF SAME DIST APPROX.
        # THIS WILL LIKELY DO.
        deltas = np.array([t_obs[i] - t_obs[i-1] for i in range(1,t_obs.shape[0])])
        unique_delta = np.unique(np.round(deltas,7))
        # Precalculate A and b prior to.
        tenors = t_mat_grid[:, 0] - t_obs[0] # shape (n_mat,)
        # 1. SOLVE RICATTI (Standard)
        a_integrated, A_integrated = self.cir_solution(params, x0=x0_zcb, T=tenors)
        a_integrated, A_integrated = -a_integrated, -A_integrated

        # 2. SCALE BY TIME  So similar to yields
        A = A_integrated #/ tenors[:, None]
        a = a_integrated #/ tenors
        # Utilize that we can compute this up front.
        # Create arrays based on obs. May vary depending on cascading or not.
        phis = []
        for i in range(0,unique_delta.shape[0]):
            if self.cascading:
                phi_0 = np.zeros(self.X_dim)
                phi_0[-1] =  kappa_p[-1] * theta_p[-1] * unique_delta[i]

                phi_1 = np.identity(self.X_dim) + np.identity(self.X_dim) * (-kappa_p) *  unique_delta[i]
                for j in range(self.X_dim):
                    if j + 1 < self.X_dim:
                        phi_1[j, j+1] = theta_p[j]*kappa_p[j] *  unique_delta[i]
                phis.append([phi_0,phi_1])
            else:
                phi_0 =  (np.identity(kappa_P_diag.shape[0])-expm(-kappa_P_diag * unique_delta[i])) @ theta_p
                phi_1 =  expm(-kappa_P_diag * unique_delta[i])
                phis.append([phi_0,phi_1])
        for n in range(0,n_obs):
            # Special first case
            # UPDATE STEP
            Zn, vn,S_k, Xn, Pn = self.Update_step(pred_Xn,pred_Pn,A,a,Sigma,Y[n,:])
            Xn = np.maximum(Xn,1e-10) # Truncate Xn
            # punish hashly if Xn below zero (mainly i).

            if result == True:
                Xn_out[n,:] = Xn
                Zn_out[n,:] = A @ Xn + a
                Pn_out[n,:,:] = Pn


            # Use log determinant
            det_S = np.linalg.det(S_k)
            # Some fallback / numerical fixes
            if (np.isnan(det_S)) | (det_S < 1e-12) :
                S_inv = np.linalg.pinv(S_k)
            else:
                S_inv = np.linalg.inv(S_k)
            # ---- 3. safe inverse / logdet ----
            sign, logdet = np.linalg.slogdet(S_k)
            if sign <= 0:
                # Introduce slight bias.
                logdet = np.log(np.abs(det_S) + 1e-12)


            log_likelihood[n] = - 0.5 * (S_k.shape[0] * np.log(2*np.pi) +logdet +
                                        vn.T @ S_inv @ vn
            )

            # Use CIR Variance for this (not going to need it)
            if (n < n_obs - 1): # Not sensible to predict further.
                Delta = t_obs[n+1] - t_obs[n]

                # Works for Uncorrelated and indep noise.
                if self.cascading:
                    Q_t = np.diag(sigma**2 * Xn) * Delta
                else:
                    Q_t = (sigma**2 * theta_p * (1-np.exp(-(kappa_p) * Delta))**2 / (2 * (kappa_p)) +
                            Xn * sigma**2 * (np.exp(-(kappa_p) * Delta) - np.exp(-2*(kappa_p) * Delta))/ (kappa_p))
                    Q_t = np.diag( Q_t.flatten())

                delta_curr = np.round(Delta,7)
                delta_idx = np.argmax(delta_curr == unique_delta)
                phi_0 = phis[int(delta_idx)][0]
                phi_X = phis[int(delta_idx)][1]
                # The prediction step effectively assumes normality.
                pred_Xn, pred_Pn = self.Prediction_step(Xn,Pn,phi_X,phi_0,Q_t)
                # Ensure positivity.
                pred_Xn = np.maximum(pred_Xn, 1e-10)
        if result == True:
            return  log_likelihood, Xn_out,Zn_out, Pn_out
        else:
            return -np.sum(log_likelihood)

    # Covariance function
    def get_cond_var(self, K, M0, Delta):
        Lambda, E = np.linalg.eig(K) # Just eigenvalues as array, columns are eigenvectors.

        S_bar = np.linalg.inv(E) @ M0 @ np.linalg.inv(E).T

        dim = K.shape[0] # assume every matrix is of same size and shape.

        # Then V_Delta.
        V_Delta = np.zeros((dim,dim))
        V_Delta_inf = np.zeros((dim,dim))

        for i in range(0,dim):
            for j in range(0,dim):
                exp_factor = (1 - np.exp(-(Lambda[i] + Lambda[j]) * Delta))
                V_Delta[i,j] = S_bar[i,j] * exp_factor / (Lambda[i] + Lambda[j])
                V_Delta_inf[i,j] = S_bar[i,j] / (Lambda[i] + Lambda[j])

        Var = E @ V_Delta @ E.T
        Var_inf = E @ V_Delta_inf @ E.T

        return Var, Var_inf


    ### Standar error calculation. Run outside as comp needed for each
    def kalman_SE(self, params, t_obs, t_mat_grid, Y,result,eps=1e-6):
        se = np.zeros((params.shape[0],params.shape[0]))
        ll_0, *_ = self.Kalman(params, t_obs, t_mat_grid, Y,result)
        n = ll_0.shape[0]
        g = np.zeros((n, params.shape[0]))
        for j in range(params.shape[0]):
            e = np.zeros(params.shape[0])
            e[j] = eps
            # Run filter. Yields Shifted ll for each obs date.
            # In this model return - f as f already returns neg log lik.
            right_end, *_ =  self.Kalman(params+e, t_obs, t_mat_grid, Y,result)
            left_end, *_  =  self.Kalman(params-e, t_obs, t_mat_grid, Y,result)
            g[:,j] = (right_end- left_end) / (2*eps)
            # Then compute standard error exact for kalman filter

        # Then loop over dates to compute SE
        for date in  range(n):
            se += np.outer(g[date,:], g[date,:])
        # SE matrix is inverted cov estimate. Asymptotics
        se = np.linalg.pinv(se) 
        se_vec = np.sqrt(np.diag(se))
        return se_vec


    def feller_constraint(self,params):
        d = self.X_dim
        # unpack
        kappa     = params[0:d]
        theta     = params[d:2*d]
        sigma     = params[2*d:3*d]
        lambda1   = params[3*d:4*d]
        sigma_err = params[-1]

        kappa_p,  theta_p = self.build_P_drift(lambda1,np.zeros_like(lambda1),kappa,theta)

        cons = []

        # latent CIR Feller: 2*kappa*theta - sigma^2 >= 0
        # combine feller conditions.
        prod_min = np.minimum(kappa*theta,kappa_p*theta_p)
        # if self.cascading:
        #     feller = 2*prod_min[-1] - sigma[-1]**2  # vector length d
        #     cons.append(np.asarray(feller).flatten())
        # else:
        # Feller is only necessary for last factor as in LHC!!!
        feller = 2*prod_min[-1] - sigma[-1]**2  # vector length d
        cons.append(np.asarray(feller).flatten())
        # Constraint on lambda.
        g3 = kappa - lambda1
        cons.append(g3)
        # g4 = lambda1 + kappa*theta
        # cons.append(g4)
        # add identification constraint. Kappas decreasing.
        # if (self.X_dim>1) & (self.cascading == False):
        # g5 = np.asarray(kappa[:-1]-kappa[1:]+1e-3).flatten() # kappai <= kappa_{i-1}
        # cons.append(np.asarray(g5).flatten())

        # if self.X_dim>1:
        #     g5 = np.asarray(theta[:-1]-theta[1:]+1e-3).flatten() # kappai <= kappa_{i-1}
        #     cons.append(np.asarray(g5).flatten())
        # We will do the same for theta.
        # if  (self.X_dim>1) & (self.cascading == False):
        # g5 = np.asarray(lambda1[1:]-lambda1[:-1]+1e-3).flatten() # kappai <= kappa_{i-1}
        # cons.append(np.asarray(g5).flatten())
        # concatenate both
        return np.concatenate(cons)  # length 2*d

    # Optimizer function.
    def run_kalman_filter(self, t_obs,t_mat_grid,Y,seed=1000):
        self.set_params(params=None,seed=seed)

        # Get initial values. Z
        x0 = np.concatenate([self.kappa, self.theta, self.sigma,
                             self.lambda1,self.sigma_err])
        x0 = x0.flatten()

        # Try a different optimizer than nelder mead
        d = self.X_dim  # number of factors
        # Kappa and theta may be slightly larger than 1 presumably.
        bounds = (
            [(1e-6, 2)] * d +     # kappa
            [(1e-6, 1)] * d +     # theta - bound by 1 to keep down
            [(1e-6, 2)] * d +     # sigma (assuming per-factor vol)
            [(-0.6, 0.6)] * d  +         # lambda_p restrict it
            [(1e-5, 0.05)]         # sigma_err
        )
        nonlinear_constraint = NonlinearConstraint(self.feller_constraint, 0, np.inf)
        # Scaling down optimization here as it takes quite a long time.
        result = differential_evolution(
            func=self.Kalman,
            bounds=bounds,
            constraints=(nonlinear_constraint,),
            # args=(t_obs[::5], t_mat_grid[:,::5], Y[::5,:]),
            args=(t_obs, t_mat_grid, Y),
            strategy='best1bin',        # more exploitative
            popsize=5 * self.X_dim,     # more modest
            maxiter=2000,               # allow more evolution
            mutation=(0.3, 0.8),        # narrower, less wild
            recombination=0.7,          # less aggressive mixing
            tol=1e-3,   
            # workers=1,
            # updating='immediate',
            workers=-1,
            updating='deferred',
            polish=False
        )

        # If DE failed or hit constraint penalties, bail early
        if result.fun >= 1e12:
            return result.x, 0, 0, 0

        # --- Local Refinement ---

        polish_result = minimize(
        fun=self.Kalman,
            x0=result.x,
            args=(t_obs, t_mat_grid, Y),
            method='trust-constr',
            bounds = bounds,
            constraints=(nonlinear_constraint,),
            options={
                'disp': False,
                'maxiter': 75,
                'gtol': 1e-4,   
                'xtol': 1e-6,   
                'barrier_tol': 1e-3, 
            }
        )
        # polish_result = result
        params = polish_result.x
        self.kalman_obj  = polish_result.fun
        # params = result.x
        # self.kalman_obj  = result.fun
        # Run and return solution
        log_likelihood,Xn,Zn,Pn = self.Kalman(params,t_obs, t_mat_grid, Y,True)
        se = self.kalman_SE(params,t_obs=t_obs, t_mat_grid=t_mat_grid, Y=Y,result=True)
        return np.sum(log_likelihood), params , Xn,Zn,Pn,se

    def run_n_kalman(self, t_obs,t_mat_grid,Y,base_seed=1000,n_restarts=5):
        # Define grid of values.
        current_objective = 1e10 #very high objective.
        for i in range(n_restarts):
            print(f"Optimization {i+1}")
            # Test several random points.
            optim_params = self.run_kalman_filter(t_obs,t_mat_grid, Y,seed=base_seed+i)
            # Test new constraints

            if (self.kalman_obj < current_objective):
                print(f"New optimal parameters at iteration {i+1}.")
                current_objective = self.kalman_obj
                out_params = optim_params
                # Assuming we do get anything for optimizing.
                Xn_out,Zn_out,Pn_out = self.Kalman(out_params,t_obs, t_mat_grid, Y,True)

        return out_params, Xn_out,Zn_out,Pn_out



    # Simulation will likely also be th eway to go about expression in Filipovic (tedious)
    def simulate_intensity(self, lambda0,T,M,scheme,seed=1000, measure = 'Q'):
        if measure == 'Q':
            kappa = self.kappa
            theta = self.theta
        elif measure == 'P':
            kappa = self.kappa_p
            theta = self.theta_p
        # Do baseline calculations
        delta = T / M
        X_dim = lambda0.shape[0]
        T_return = np.array([0] + [delta*k for k in range(1,M+1)])
        path = (np.ones(shape = (X_dim,M + 1))* lambda0.reshape((X_dim,1)) ).T # to include zero.
        W = norm.rvs(size = (X_dim*M),random_state=seed).reshape((M,X_dim)) # simulate at beginning - faster!
        # Creat Matrices
        if self.cascading:
            K1 = np.diag(-kappa.flatten())
            for i in range(self.X_dim):
                if i + 1 < self.X_dim:
                    K1[i, i+1] = kappa[i] * theta[i]
            K0 = np.zeros(self.X_dim)
            K0[-1] = kappa[-1] * theta[-1]
        else:
            kappa_mat = np.diag(kappa.flatten())
            theta_vec = theta

        sigma_mat = np.diag(self.sigma.flatten())
        if scheme == "Euler":
            for i in range(1,M+1):
                path_prev = np.maximum(path[i-1,:],  1e-8) # Clip to have valid sqrt
                if self.cascading:
                    mu_t = K0 +  K1 @ path_prev
                else:
                    mu_t = kappa_mat @ (theta_vec - path_prev)

                sigma_t = sigma_mat *  np.sqrt(path_prev)
                path[i,:] = path_prev + delta*mu_t +  np.sqrt(delta) * sigma_t @ W[i-1,:]
        elif scheme == "Milstein":
            for i in range(1,M+1):
                path_prev = np.maximum(path[i-1,:], 1e-8) # Clip to have valid sqrt
                if self.cascading:
                    mu_t = K0 +  K1 @ path_prev
                else:
                    mu_t = kappa_mat @ (theta_vec - path_prev)

                sigma_t = sigma_mat *  np.sqrt(path_prev)
                sigma_prime_t = sigma_mat * 1/(2*np.sqrt(path_prev))
                path[i,:] = path_prev+ delta*mu_t +  np.sqrt(delta) * sigma_t @ W[i-1,:]+ 1/2 * delta* sigma_prime_t @ sigma_t @ (W[i-1,:]**2-1)
        # elif scheme == "Exact":
        #     # TODO: Correct this.
        #     for i in range(1,M+1):
        #         path_prev = np.maximum(path[i-1,:], 0) # Clip to have valid sqrt
        #         k = 4 * self.kappa * self.theta / (self.sigma**2)
        #         l = 4 * self.kappa * np.exp(-self.kappa * T_return[i]) / (self.sigma**2 * (1-np.exp(-self.kappa *T_return[i]))) * path[i-1]
        #         factor = self.sigma**2 * (1 - np.exp(-self.kappa * T_return[i])) / (4*self.kappa)
        #         path[i] = factor * ncx2.rvs(df = k, nc = l, random_state = seed)

        return T_return, path



    # MC pricing.
    def get_cdso_pric_MC(self,params,t,t0,t_M,strikes,X0,N,M,seed=1000):
        # N prices are comuted and averaged MC
        N_strikes = strikes.shape[0]
        prices = np.zeros(shape = (N,N_strikes))
        prices_MC_hist =  np.zeros(shape = (N,N_strikes))
        # Solve Ricatti equations prior to loop - saves time, same for all.
        # precompute up to max maturity requested.
        T_max = t_M - t
        interp_struct = self.precompute_affine_grid(params, T_max, N_grid= 10000+1, rho=1)
        # With structure, ready to price options.
        for i in range(N):
            # Get default intensity process.
            T_return,X_t = self.simulate_intensity(X0,t0,M,scheme = 'Milstein',
                                                   seed=seed,measure = 'Q')
            # lambda_t is the sum of latent states..
            # lambda_t = np.sum(X_t,axis=1)
            lambda_t = X_t[:,0]

            # Compute prob of default at time t0
            deltas = np.array([T_return[i]-T_return[i-1] for i in range(1,M+1)])
            Lambda = np.cumsum(lambda_t[1:]*deltas)
            # The survival process is in turn.
            St = np.exp(-Lambda) # And this starts close to 1 contrarily to Yt.
            # Determine if default or not at t0. If lambda>E\simEXPo(1) option payoff is zero.
            # E = expon.rvs(random_state = seed)
            U = uniform.rvs(random_state = seed)

            if St[-1] <= U:
                prices[i] = 0
            # Else - begin to compute prices.
            else:
                X_t0 = X_t[-1,:]
                # prot = self.calc_protection_leg(params, t0,t0, t_M, X_t0)
                # # Quick fix due to way its written
                # I1 = self.calc_coupon_leg(params,t0, t0, t_M, X_t0)
                # I2 = self.calc_accrual_leg(params,t0, t0, t_M, X_t0)
                # Value_CDS_old = prot - strikes * (I1 + I2)
                #  params, t, t0, t_mat, lambda_t, interp_struct, tau_fine_per_coupon
                prot = self.calc_protection_leg_fast(params, t0,t0, np.array([t_M]), X_t0,interp_struct, tau_fine_per_coupon=80)
                # Quick fix due to way its written
                I1 = self.calc_coupon_leg_fast(params,t0, t0, np.array([t_M]), X_t0,interp_struct, tau_fine_per_coupon=80)
                I2 = self.calc_accrual_leg_fast(params,t0, t0, np.array([t_M]), X_t0,interp_struct, tau_fine_per_coupon=40)
                Value_CDS =  prot - strikes * (I1 + I2)
                # Discount back:
                # Note still an option, so only enter if positive.
                prices[i,:] = np.exp(-self.r * (t0 - t)) * np.maximum(Value_CDS,0)
            # Achieve a running mean also for convergence assessment.
            prices_MC_hist[i, :] = np.mean(prices[:i+1, :], axis=0)
            seed += 1
        print(f'CDSO price at Done')

        price_MC = np.mean(prices,axis=0)

        return prices_MC_hist,price_MC


    # Simulate digital Barrier in the model.
    def get_digital_barrier_price_MC(self,params,t,t0,t_M,T,barriers,X0,N,M,seed=1000):
        '''
        t: Time to price at
        t0: Start of CDS.
        t_M: Maturity of CDS
        T: Maturity of option. Needs to satisfy T<t_m
        '''
        # N prices are comuted and averaged MC
        N_strikes = barriers.shape[0]
        prices = np.zeros(shape = (N,N_strikes))
        prices_MC_hist = np.zeros(shape = (N,N_strikes))
        T_max = t_M - t
        interp_struct = self.precompute_affine_grid(params, T_max, N_grid= 10000+1, rho=1)

        for i in range(N):
            # Get Latent states. Simulate path of CDS till option mat.
            T_return,X_t = self.simulate_intensity(X0,T,M,scheme = 'Milstein',
                                                   seed=seed,measure='Q')
            # Compute prob of default at time T
            lambda_t = X_t[:,0] # First state is deefault intensity.
            # Compute prob of default at time T
            deltas = np.array([T_return[i]-T_return[i-1] for i in range(1,M+1)])
            Lambda = np.cumsum(lambda_t[1:]*deltas)
            # Determine if default or not at t0. If lambda>E\simEXPo(1) option payoff is zero.
            St = np.exp(-Lambda)
            # E = expon.rvs(random_state = seed)
            U = uniform.rvs(random_state = seed)

            # Should be zero, but depends on barrier. Default happens at some point below..
            default_event =  St <= U
            
            # Get model implies CDS.
            CDS_sim = np.zeros(Lambda.shape)
            for n in range(CDS_sim.shape[0]):
                CDS_sim[n]  = self.calc_CDS_fast_numba(T_return[n],t0, t_M, X_t[n,:],
                                                interp_struct)


                # Break loop if default happens at date
                if default_event[n]:
                    break
            if np.any(default_event):
                # In this instance, default has happened as some point. Find index.
                idx = np.argmax(default_event==1)
                # Get path maximum up to the point:
                max_cds_to_default = np.max(CDS_sim[:idx+1])
                b_idx = np.where(max_cds_to_default >= barriers)
                # In this case, there is a payoff of 1, discount back from expiry(pay date) to today
                prices[i,b_idx] = np.exp(-self.r * (T - t))

            # Else - no default happened, but same logic as before.
            else:
                # Get path maximum up to expiry:
                max_cds_to_default = np.max(CDS_sim)
                b_idx = np.where(max_cds_to_default >= barriers)
                # In this case, there is a payoff of 1, discount back from expiry(pay date) to today
                prices[i,b_idx] = np.exp(-self.r * (T - t))
            prices_MC_hist[i, :] = np.mean(prices[:i+1, :], axis=0)
            seed += 1
        print(f'Digital Done')

        price_MC = np.mean(prices,axis = 0)


        return prices_MC_hist,price_MC

    # Simulate digital Barrier in the model.
    def get_lookback_price_MC(self,params,t,t0,t_M,T,X0,N,M,seed=1000):
        '''
        t: Time to price at
        t0: Start of CDS.
        t_M: Maturity of CDS
        T: Maturity of option. Needs to satisfy T<t_m
        '''
        # N prices are comuted and averaged MC
        prices = np.zeros(shape = N)
        prices_MC_hist = np.zeros(shape = N)
        cds_min =  np.zeros(shape = N)
        T_max = t_M - t
        interp_struct = self.precompute_affine_grid(params, T_max, N_grid= 10000+1, rho=1)

        for i in range(N):
            # Get Latent states. Simulate path of CDS till mat.
            T_return,X_t = self.simulate_intensity(X0,T,M,scheme = 'Milstein',
                                                   seed=seed,measure='Q'
            )
            # Compute prob of default at time T
            # lambda_t = np.sum(X_t,axis=1)
            lambda_t = X_t[:,0] # First state is deefault intensity.

            # Compute prob of default at time T
            deltas = np.array([T_return[i]-T_return[i-1] for i in range(1,M+1)])
            Lambda = np.cumsum(lambda_t[1:]*deltas)
            # Determine if default or not at t0. If lambda>E\simEXPo(1) option payoff is zero.
            St = np.exp(-Lambda)
            # E = expon.rvs(random_state = seed)
            U = uniform.rvs(random_state = seed)

            # Should be zero, but depends on barrier. Default happens at some point below..
            default_event =  St <= U
            # Should be zero, but depends on barrier. Default happens at some point below..
            # default_event =  Lambda >= E
            # Get model implies CDS.
            CDS_sim = np.zeros(Lambda.shape)
            for n in range(CDS_sim.shape[0]):
                CDS_sim[n]  = self.calc_CDS_fast_numba(T_return[n],t0, t_M, X_t[n,:],
                                                interp_struct)
                # Break loop if default happens at date
                if default_event[n]:
                    break

            if np.any(default_event):
                prices[i] = 0
                # set min to zero here
                cds_min[i] = 0  # np.min(CDS_sim[:i])
            # Else - no default happened
            else:
                # Get path minimum of CDS:
                cds_min[i]  = np.min(CDS_sim)
                prices[i] = np.exp(-self.r * (T - t)) * (CDS_sim[-1] - cds_min[i] )
            prices_MC_hist[i] = np.mean(prices[:i+1])
            seed += 1

        print(f'Lookback at Done')
        price_MC = np.mean(prices)

        return prices_MC_hist,price_MC,cds_min

    ##### The Option inversion formula (can be used to eurpean like options.)
    def G_transform(self,y,a,b,Xt,T):
        params = np.concatenate([self.kappa, self.theta, self.sigma,self.kappa_p, self.theta_p,self.sigma_err])
        first_term = self.Laplace_Transform(params,Xt,T,a)
        Laplace_fixed = lambda v: self.Laplace_Transform(params,Xt,T,a + 1j * v * b)
        lower_G = 1e-6  # small shift away from 0
        upper_G = 100
        Laplace_int, _ = quad(
            lambda v: np.imag(Laplace_fixed(v) * np.exp(-1j * v * y)) / v,
            lower_G, upper_G,
            limit=200,
            epsabs=1e-12,
            epsrel=1e-12
        )

        return np.real(first_term/2 - Laplace_int/np.pi)


    def cir_solution_T(self,cir_params,T,rho=1):
        x0 = np.zeros(self.X_dim)
        kappa,theta,sigma1,lambda_i,sigma_err = self.unpack_params(cir_params)
        gamma = np.sqrt(kappa**2 + 2*sigma1**2*rho)
        beta_nom1 =  (- 2 * rho * gamma*(np.exp(gamma*T)) +
                        gamma*x0 * np.exp(gamma * T) * (gamma - kappa))

        beta_denom = (2 * gamma +
                        (gamma + kappa - x0 * sigma1**2) * (np.exp(gamma * T) - 1))


        term1 = beta_nom1 / beta_denom

        beta_nom2 = (- 2 * rho * (np.exp(gamma*T)-1) +
                        x0 * np.exp(gamma * T) * (gamma - kappa) +
                        x0 * (gamma + kappa)) *  (gamma + kappa - x0 * sigma1**2) *gamma* (np.exp(gamma * T))
        beta_denom2 = (2 * gamma +
                        (gamma + kappa - x0 * sigma1**2) * (np.exp(gamma * T) - 1))**2

        term2 = beta_nom2 / beta_denom2

        beta_T = term1 - term2

        # get alpha, alo verify above.
        alpha_log_nom = (2 * gamma * np.exp((gamma + kappa ) * T / 2))

        alpha_log_denom = (2 * gamma +
                            (gamma + kappa - x0 * sigma1**2)*(np.exp(gamma * T)-1))

        alpha_T = 2 * kappa * theta / sigma1**2 * (
            alpha_log_denom/alpha_log_nom * (
                gamma * (gamma + kappa) / alpha_log_denom -
                alpha_log_nom * (gamma + kappa - x0*sigma1**2) * gamma * np.exp(gamma*T) /alpha_log_denom
            )
        )

        return alpha_T, beta_T


## Adding optimized code for kalman filter and cds conversinos

    def precompute_affine_grid(self, params, T_max, N_grid=500, rho=1):
        # tau grid from 0 to T_max (include 0)
        tau_grid = np.linspace(0.0, float(T_max), int(N_grid))
        # compute once: use x0 = zeros (as your Laplace calls do)
        x0 = np.zeros(self.X_dim)
        # cir_solution expects T possibly array-like
        alpha_grid, beta_grid = self.cir_solution(params, x0, tau_grid, rho=rho)
        # shapes: alpha_grid (len(tau_grid),), beta_grid (len(tau_grid), X_dim) in your implementation
        # ensure arrays are numpy arrays
        alpha_grid = np.asarray(alpha_grid).reshape(len(tau_grid), -1).squeeze()
        beta_grid = np.asarray(beta_grid)
        # derivatives
        alpha_x_grid, beta_x_grid = self.cir_derivatives(params, x0, tau_grid, rho=rho)
        alpha_x_grid = np.asarray(alpha_x_grid).reshape(len(tau_grid), -1).squeeze()
        beta_x_grid = np.asarray(beta_x_grid)

        # Build interpolants for alpha (scalar), beta (vector), alpha_x (scalar), beta_x (matrix)
        # interp1d expects axis 0 = tau
        # kind = 'cubic' if len(tau_grid) >= 8 else 'linear'
        kind = 'cubic'
        # interpolate linearly i.e.
        alpha_interp = interp1d(tau_grid, alpha_grid, kind=kind, axis=0, fill_value='extrapolate', assume_sorted=True)
        alpha_x_interp = interp1d(tau_grid, alpha_x_grid, kind=kind, axis=0, fill_value='extrapolate', assume_sorted=True)
        beta_interp = interp1d(tau_grid, beta_grid, kind=kind, axis=0, fill_value='extrapolate', assume_sorted=True)
        beta_x_interp = interp1d(tau_grid, beta_x_grid, kind=kind, axis=0, fill_value='extrapolate', assume_sorted=True)

        return {
            'tau_grid': tau_grid,
            'alpha_grid': alpha_grid,
            'beta_grid': beta_grid,
            'alpha_x_grid': alpha_x_grid,
            'beta_x_grid': beta_x_grid,
            'alpha_interp': alpha_interp,
            'beta_interp': beta_interp,
            'alpha_x_interp': alpha_x_interp,
            'beta_x_interp': beta_x_interp
        }


    def interpolate_laplace(self, interp_struct, lambda_t, tau):
        # Function to get laplace given interpolation
        # Ensure lambda_t as 1D vector of length X_dim
        lam = np.asarray(lambda_t).ravel()
        alpha_vals = interp_struct['alpha_interp'](tau)         # shape (len(tau),)
        beta_vals = interp_struct['beta_interp'](tau)           # shape (len(tau), X_dim)

        expo = alpha_vals  + beta_vals @ lam
        return np.exp(expo)


    def interpolate_derivates(self, interp_struct, lambda_t, tau):
        lam = np.asarray(lambda_t).ravel()
        alpha_x_vals = interp_struct['alpha_x_interp'](tau)     # (len(tau),)
        beta_x_vals = interp_struct['beta_x_interp'](tau)       # (len(tau), X_dim)
        # The einsum is to get it for each lamba
        return alpha_x_vals + beta_x_vals @ lam  # (len(tau),)


    # Vectorized fast legs (replace your old calc_* with these)
    def calc_coupon_leg_fast(self, params, t, t0, t_mat, lambda_t, interp_struct, tau_fine_per_coupon=40):
        # payment dates grid
        t_grid = np.arange(t0, float(t_mat) + 1e-12, self.tenor)
        total = 0.0
        # loop over payment dates to the coupon leg.
        for j in range(1, len(t_grid)):
            a = t_grid[j-1]
            b = t_grid[j]
            # Evaluate laplace
            laplace = self.interpolate_laplace(interp_struct, lambda_t, np.array([b-t]))
            disc = np.exp(-self.r * (b-t))
            disc_exp = disc * laplace
            # Add sum
            total += (b - a) * disc_exp 
        return np.array([total])


    def calc_protection_leg_fast(self, params, t, t0, t_mat, lambda_t, interp_struct, tau_fine_per_coupon=80):
        # only function of time to maturity -> not payment dependnyasaccruallg
        tau_start = float(t0 - t)
        tau_end = float(t_mat - t)
        # choose number of points proportional to interval length if desired
        N = max(200, int((tau_end - tau_start) * tau_fine_per_coupon))
        tau = np.linspace(tau_start, tau_end, N)
        # intrpolat.
        disc = np.exp(-self.r * tau)
        deriv = self.interpolate_derivates(interp_struct, lambda_t, tau)   # vector
        LT_vals = self.interpolate_laplace(interp_struct, lambda_t, tau)
        integrand = (1.0 - self.delta) * disc * deriv * LT_vals
        prot_val = np.trapz(integrand, tau)
        return prot_val


    def calc_accrual_leg_fast(self, params, t, t0, t_mat, lambda_t, interp_struct, tau_fine_per_coupon=40):
        t_grid = np.arange(t0, float(t_mat) + 1e-12, self.tenor)
        if len(t_grid) < 2:
            return 0.0
        total = 0.0
        # Approximate integral for each payment dat
        for j in range(1, len(t_grid)):
            a = t_grid[j-1]
            b = t_grid[j]
            # Time to mat for each entry stratisfied on grid. Used for disc.
            tau_local = np.linspace(a - t, b - t, tau_fine_per_coupon)
            # Build grid og values in t_i-1,t_i 
            u_local = tau_local + t
            # Correct up till default date
            default_frac = u_local - a   # vector
            # Discount terms for each subsum of paymaent dates
            disc = np.exp(-self.r * tau_local)
            # Evaluate drivative at each point here:
            deriv = self.interpolate_derivates(interp_struct, lambda_t, tau_local)
            # Evaluate laplace similarly using interpolation.
            LT_vals = self.interpolate_laplace(interp_struct, lambda_t, tau_local)
            integrand = disc * default_frac * deriv * LT_vals
            total += np.trapz(integrand, tau_local)
        return total


    def calc_CDS_fast(self, params, t,t0, t_mat, lambda_t, interp_struct):
        prot = self.calc_protection_leg_fast(params, t,t0, t_mat, lambda_t, interp_struct)
        I1 = self.calc_coupon_leg_fast(params, t,t0, t_mat, lambda_t, interp_struct)[0]
        I2 = self.calc_accrual_leg_fast(params, t,t0, t_mat, lambda_t, interp_struct)
        # avoid divide by zero
        denom = I1 + I2
        if denom == 0:
            return 0.0
        return prot / denom
        # prot_val = self.calc_protection_leg(params,t,t0,t_mat, lambda_t)
        # I1 = self.calc_coupon_leg(params,t,t0,t_mat, lambda_t)
        # I2 = self.calc_accrual_leg(params,t,t0,t_mat, lambda_t)

    def cds_spread_fast(self, X, params, t, t_mat_grid, t0=None, interp_struct=None):
        if t0 is None:
            t0 = t
        if interp_struct is None:
            # precompute up to max maturity requested
            T_max = np.max(t_mat_grid) - t
            interp_struct = self.precompute_affine_grid(params, T_max, N_grid= 10000+1, rho=1)
        results = np.zeros_like(t_mat_grid, dtype=float)
        for i, T_mat in enumerate(t_mat_grid):
            results[i] = self.calc_CDS_fast(params, t,t0, np.array([T_mat]), X, interp_struct)
        # test = self.cds_spread(X,params, t, np.array([t_mat_grid[-1]]),t0=None)
        return results


    def calc_CDS_fast_numba(self, t, t0, t_mat, X, interp):
        # utilofAItowritthisinafastmannr.
        return calc_CDS_numba(
            t, t0, t_mat, X,
            interp['tau_grid'],
            interp['alpha_grid'],
            interp['beta_grid'],
            interp['alpha_x_grid'],
            interp['beta_x_grid'],
            self.r, self.delta, self.tenor
        )
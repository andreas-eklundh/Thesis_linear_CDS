import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, Bounds

from numba import njit, float64, int64
from Models.ATSMGeneral.ATSM import ATSM
from Models.BaselineCIR_alternative.CIR_numba import calc_cds
from Models.BaselineCIR_alternative.Gamma_solver import DeterministicGamma
from scipy.stats import norm, ncx2, gamma, expon
from scipy.integrate import quad
from scipy.linalg import expm
from numba.experimental import jitclass
from scipy.optimize import differential_evolution


## Class to actually call
class CIRIntensity():
    def __init__(self, r, delta, tenor,X_dim=1):
        self.r = r
        self.delta = delta
        self.tenor = tenor
        self.X_dim = X_dim
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
            self.kappa = rng.uniform(0.2, 0.8, size=(X_dim,))
            self.theta = rng.uniform(0.01, 0.1, size=(X_dim,))

            # Initiate MPR specification. Use positivity bound for positive.
            self.lambda_i = np.zeros(X_dim) 
            for i in range(X_dim):
                self.lambda_i[i] = rng.uniform(-self.theta[i]*self.kappa[i], self.kappa[i], 1) 

            # initialise all positive.
            self.kappa_p = self.kappa - self.lambda_i
            self.theta_p = (self.theta *self.kappa + self.lambda_i ) / self.kappa_p

            # Set sigma in feller grid.
            # Sigma to be in minimum of feller conds.
            feller_min = np.minimum(np.sqrt(2*self.kappa*self.theta),
                                    np.sqrt(2*self.kappa_p*self.theta_p))
            self.sigma = rng.uniform(0.001, feller_min, size=(X_dim,))
            
            self.sigma_err = rng.uniform(0.001, 0.04, size=(1,))


        else:
            # Else, asusming some paramter tuning, set then here.
            self.kappa, self.theta, self.sigma,self.lambda_i,self.sigma_err = self.unpack_params(params)
            self.kappa_p = self.kappa - self.lambda_i
            self.theta_p = (self.theta *self.kappa + self.lambda_i) / self.kappa_p
    
    def unpack_params(self,params):
        X_dim = self.X_dim
        kappa, theta, sigma = params[:X_dim],params[X_dim:2*X_dim],params[2*X_dim:3*X_dim]
        lambda_i, sigma_err = params[3*X_dim:4*X_dim], np.array([params[-1]])
        return kappa,theta,sigma,lambda_i,sigma_err
    #### Solve affine equations.

    # For reference, also the solution in Lando (2004).
    # Potentially more handystable (numba). Default is rho=1 as it will be the one we use (maybe?)

    def cir_solution(self,params,x0,T,rho=1,corr=False):
        # Local copies of kappa, theta to minimize code. Rename to comply with Lando.
        kappa,theta,sigma1,lambda_i,sigma_err = self.unpack_params(params)
        gamma = np.sqrt(kappa**2 + 2*sigma1**2*rho)
        # If x0 is one dimensional (intensity), use Lando forthetalas
        
        if (corr == False) & (np.all(np.isreal(x0))):
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
        else:
            # Build matrices for Afine term structure models.
            K0 = np.identity(self.X_dim) * kappa @ theta
            K1 = - np.identity(self.X_dim) * kappa
            H0 = np.zeros((self.X_dim,self.X_dim ))
            H1 = np.zeros((self.X_dim,self.X_dim,self.X_dim))
            for i in range(self.X_dim):
                H1[i,i,i] = sigma1[i]**2
            # Just assume first factor is default intensity.
            rho1 = np.ones(self.X_dim)
            # rho1[0] = 1

            atsm = ATSM(K0,K1,H0,H1,rho0=0,rho1=rho1)
            atsm.solve_ODE_system(x0,0,T)

            return atsm.alpha,atsm.beta.T

    def cir_derivatives(self,params,x,T,rho=1, corr=False):
        # Can work in 
        kappa,theta,sigma1,lambda_i,sigma_err = self.unpack_params(params)
        gamma = np.sqrt(kappa**2 + 2*sigma1**2*rho)
        if corr == False:
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
            expectation = self.Laplace_Transform(params, lambda_t.T, t_grid[t_idx] - t)
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
        integrand = lambda u: (np.exp(-self.r * (u-t)) * self._get_default_grid(u,t_grid) *  
            (self.cir_derivatives(params,x,u-t)[0] + 
            self.cir_derivatives(params,x,u-t)[1] @ lambda_t.T) *
            self.Laplace_Transform(params,lambda_t.T, u - t)
        )

        Al_val = 0.0
        for i in range(len(t_grid) - 1):
            a, b = t_grid[i], t_grid[i + 1]
            Al_val += quad(integrand, a, b, epsabs=1e-8, epsrel=1e-8, limit=100)[0]
        return Al_val

    # Protection leg:
    def calc_protection_leg(self,params,t,t0,t_mat, lambda_t):
        x = np.zeros(self.X_dim)

        integrand = lambda u: ((1- self.delta)*np.exp(-self.r * (u-t)) * (
            self.cir_derivatives(params,x,u-t)[0] + 
            self.cir_derivatives(params,x,u-t)[1] @ lambda_t.T)*
            self.Laplace_Transform(params,lambda_t.T, u - t)
            )
        prot_val, _ = quad(integrand,t0,t_mat,epsabs=1e-9, epsrel=1e-9)

        return prot_val



    def calc_CDS(self,params,t,t_mat, lambda_t,t0=None):
        # If no t0 provided, assume at inception
        if t0 == None:
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
        # print(params)
        kappa,theta,sigma,lambda_i,sigma_err = self.unpack_params(params)
        kappa_p = kappa - lambda_i
        theta_p = (theta * kappa + lambda_i ) / kappa_p
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
        beta = 2 *kappa_p / sigma**2
        # CIR values.
        X0 = alpha / beta
        P0 = alpha / beta**2
        # CIR conditional Mean and Variance based on parameters.
        L = X0.shape[0]

        Xn = np.zeros((L))
        Zn = np.zeros((n_mat))
        Pn = np.zeros((L,L))
        if result == True:
            Xn_out = np.zeros((n_obs,L))
            Zn_out = np.zeros((n_obs,n_mat))
            Pn_out = np.zeros((n_obs,L,L))
        # We want to store all predictions.
        pred_Xn  = X0
        pred_Pn  = ( np.identity(self.X_dim) * P0).reshape((self.X_dim,self.X_dim))
        Delta = t_obs[1] - t_obs[0] # Only apprx for now. Move to loop maybe.

        log_likelihood = 0
        # Run algo.
        x0_zcb = np.zeros(self.X_dim)
        kappa_P_diag = np.identity(self.X_dim) * kappa_p

        # To speed up, solve ricatti equations. Will be time homogenous.
        # Solve Ricatti Equations. Might move inside loop later - MUCH FASTER OUT HERE, IF SAME DIST APPROX.
        # THIS WILL LIKELY DO.
        a,A =  self.cir_solution(params,x0 = x0_zcb,T = t_mat_grid[:,0] - t_obs[0])
        a,A = -a,-A

        # Utilize that we can compute this up front.
        # Create arrays based on obs.
        phi_0 = (np.identity(kappa_P_diag.shape[0])-expm(-kappa_P_diag * Delta)) @ theta_p
        phi_X = expm(-kappa_P_diag * Delta)

        for n in range(0,n_obs):
            # UPDATE STEP
            Zn, vn,S_k, Xn, Pn = self.Update_step(pred_Xn,pred_Pn,A,a,Sigma,Y[n,:])
            Xn = np.maximum(Xn,1e-6) # Truncate Xn
            # punish hashly if Xn below zero (mainly i). 

            if result == True:
                Xn_out[n,:] = Xn
                Zn_out[n,:] = A @ Xn + a
                Pn_out[n,:,:] = Pn

            # Update log likelihood.
            det_S = np.linalg.det(S_k)
            if det_S < 0:
                return 1e12 #,Xn, Zn, Pn 

            # Some fallback / numerical fixes
            if (np.isnan(det_S)) | (det_S < 1e-12) :
                S_inv = np.linalg.pinv(S_k)
            else:
                S_inv = np.linalg.inv(S_k) 

            log_likelihood += - 0.5 * (S_k.shape[0] * np.log(2*np.pi) + np.log(det_S) +
                                        vn.T @ S_inv @ vn
            )

            # Use CIR Variance for this (not going to need it)
            if (n < n_obs - 1): # Not sensible to predict further.
                # Works for Uncorrelated and indep noise.
                Q_t = (sigma**2 * theta_p * (1-np.exp(-kappa_p * Delta))**2 / (2 * kappa_p) +
                        Xn * sigma**2 * (np.exp(-kappa_p * Delta) - np.exp(-2*kappa_p * Delta))/ kappa_p)
                Q_t = np.diag( Q_t.flatten())

                Delta = t_obs[n+1] - t_obs[n]
                # The prediction step effectively assumes normality. 
                pred_Xn, pred_Pn = self.Prediction_step(Xn,Pn,phi_X,phi_0,Q_t)
                # Ensure positivity.
                pred_Xn = np.maximum(pred_Xn, 1e-6)
        if result == True:
            return Xn_out,Zn_out, Pn_out
        else:
            return -log_likelihood


    def feller_constraint(self,params):
        d = self.X_dim
        # unpack
        kappa     = params[0:d]
        theta     = params[d:2*d]
        sigma     = params[2*d:3*d]
        lambda_i   = params[3*d:4*d]
        sigma_err = params[-1]

        kappa_p = kappa - lambda_i
        theta_p = (theta * kappa + lambda_i) / kappa_p
        cons = []

        # latent CIR Feller: 2*kappa*theta - sigma^2 >= 0
        # combine feller conditions.
        prod_min = np.minimum(kappa*theta,kappa_p*theta_p)
        feller = 2*prod_min - sigma**2  # vector length d
        cons.append(feller)
        # Constraint on lambda.
        
        for i in range(kappa.shape[0]):
            g3 = kappa[i] - lambda_i[i]
            cons.append(g3)
            g4 = lambda_i[i] + kappa*theta 
            cons.append(g4)
        
        # for i in range(kappa.shape[0]):
        #     g3 = kappa[i] - lambda_i[i]
        #     cons.append(g3)
            # if lambda_i[i] > 0:
            #     g3 = kappa[i] - lambda_i[i]
            #     cons.append(g3)
            # else: 
            #     g3 = kappa[i] * theta[i] + lambda_i[i]
            #     cons.append(g3)

        # concatenate both
        return np.concatenate([feller])  # length 2*d

    # Optimizer function.
    def run_kalman_filter(self, t_obs,t_mat_grid,Y,seed=1000):
        self.set_params(params=None,seed=seed)

        # Get initial values. Z
        x0 = np.concatenate([self.kappa, self.theta, self.sigma, 
                             self.lambda_i,self.sigma_err])
        x0 = x0.flatten()

        # Try a different optimizer than nelder mead
        d = self.X_dim  # number of factors

        bounds = (
            [(1e-6, 1)] * d +     # kappa
            [(1e-6, 1)] * d +     # theta
            [(1e-6, 1)] * d +     # sigma (assuming per-factor vol)
            [(-1, 1)] * d +     # lambda
            [(1e-6, 0.5)]         # sigma_err
        )
        nonlinear_constraint = NonlinearConstraint(self.feller_constraint, 0, np.inf)

        result = differential_evolution(
            func=self.Kalman,
            bounds=bounds,
            constraints=(nonlinear_constraint,),
            args=(t_obs, t_mat_grid, Y),
            strategy='best1bin',
            popsize=13,         # larger popsize -> more exploration
            maxiter=800,       # allow many generations          
            tol=1e-6,            # looser tolerance
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
            x0=result.x,  # <-- Use DE's best solution as the start
            args=(t_obs, t_mat_grid, Y),
            method='COBYLA',
            constraints=(nonlinear_constraint,), # <-- Pass the same nonlinear constraints
            options={
                'disp': True,
                'maxiter': 500  # Give it a reasonable number of iterations
            }
        )
        # not doing it. slow and yields very little.

        params = polish_result.x
        self.kalman_obj  = polish_result.fun
        # Run and return solution
        Xn,Zn,Pn = self.Kalman(params,t_obs, t_mat_grid, Y,True)

        return params , Xn,Zn,Pn
    
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

    #### Callibrate CIR++ (single dimensional). Analogous to Hull-White TSM.
    # Analogous to the Get pricing parameters in LHC
    def psi_func(self,u,params,gamma_mkt, X0,t_mats):
        params_full = np.concatenate(params, np.zeros(self.X_dim + 1))
        # Get deterministic gamma for this u
        gamma_mkt = self.gamma_fun(params,u,t_mats)
        # Get log derivative of CIR. Just t derivatives of alpha and beta.
        alpha_T,beta_T = self.cir_solution_T(params_full,u)
        du_logP = alpha_T + beta_T @ X0

        psi = gamma_mkt + du_logP

        return psi
    
    def psi_objective(self,params,X0,t_mats):
        T_mat = t_mats[-1]
        integrand = lambda u: (self.psi_func(u,params,X0)**2)
        prot_val, _ = quad(integrand,0,T_mat,epsabs=1e-9, epsrel=1e-9)

        return prot_val
    
    ## Then have a constraint that it need to be positive. 
    # def optimize_cir_pp(self,params_init,gamma_mkt,X0,t_mats,t_calc_grid):
    #     nlc = NonlinearConstraint(lambda x: self.psi_func(x,), 0, np.inf)
    #     # T_mat must here be the 
    #     result = minimize(fun=self.psi_objective,
    #                       x0=params_init,
    #                       constraints=(nlc,),
    #                       args = (T_mat,X0),
    #                       method = 'trust-constr',
    #                       options = {
    #                         "xatol": 1e-4,
    #                         "fatol": 1e-4,
    #                         "maxiter": 500,
    #                         "disp": True
    #                         }
    #                         )

    # Redefine the piecewise constant gamma func.
    def gamma_fun(self, params, u, t_mats):
        """Piecewise constant gamma(u). params length == len(t_mats)-1."""
        t_mats = np.asarray(t_mats)
        # if before first grid point
        if u < t_mats[0]:
            return 0.0
        # find index j such that t_mats[j] <= u < t_mats[j+1]
        idx = np.searchsorted(t_mats, u, side='right') - 1
        if idx < 0:
            return 0.0
        if idx >= len(params):
            # if u beyond last interval, return last param
            return params[-1]
        return params[idx]


    def nonlinear_constraints(self, T_mat,params,gamma_mkt,X0):
        """
        Vector-valued constraint function for NonlinearConstraint.
        Returns array `cons` such that cons >= 0 are feasible.
        If params can't be unpacked properly, returns a negative array (infeasible).
        """
        # expected number of constraints: g1, g2 and 2*m for g3,g4 converted -> total 2 + 2*m
        params = np.asarray(params, dtype=float)

        lambda_i,sigma,sigma_err = params[0],params[1],params[2]
        cons = []
        cons.append(self.psi_func(T_mat,params,gamma_mkt,X0))  # theta_p <= 1


        cons = np.asarray(cons, dtype=float)


        return cons



    # Simulation will likely also be th eway to go about expression in Filipovic (tedious)
    def simulate_intensity(self, lambda0,T,M,scheme,seed=1000, measure = 'Q'):
        if measure == 'Q':
            theta = self.theta
            kappa = self.kappa
        elif measure == 'P':
            theta = self.theta_p
            kappa = self.kappa_p
        # Do baseline calculations
        delta = T / M
        X_dim = lambda0.shape[0]
        T_return = np.array([0] + [delta*k for k in range(1,M+1)])
        path = (np.ones(shape = (X_dim,M + 1))* lambda0.reshape((X_dim,1)) ).T # to include zero.
        W = norm.rvs(size = (X_dim*M),random_state=seed).reshape((M,X_dim)) # simulate at beginning - faster!
        # Creat Matrices
        kappa_mat = np.diag(kappa.flatten())
        theta_vec = theta
        sigma_mat = np.diag(self.sigma.flatten())
        if scheme == "Euler":
            for i in range(1,M+1):
                path_prev = np.maximum(path[i-1,:], 0) # Clip to have valid sqrt
                mu_t = kappa_mat @ (theta_vec - path_prev)
                sigma_t = sigma_mat *  np.sqrt(path_prev)
                path[i,:] = path_prev + delta*mu_t +  np.sqrt(delta) * sigma_t @ W[i-1,:]
        elif scheme == "Milstein":
            for i in range(1,M+1):
                path_prev = np.maximum(path[i-1,:], 0) # Clip to have valid sqrt
                mu_t = kappa_mat @ (theta_vec - path_prev)
                sigma_t = sigma_mat *  np.sqrt(path_prev)
                sigma_prime_t = sigma_mat * 1/(2*np.sqrt(path_prev))
                path[i,:] = path_prev+ delta*mu_t +  np.sqrt(delta) * sigma_t @ W[i-1,:]+ 1/2 * delta* sigma_prime_t @ sigma_t @ (W[i-1,:]**2-1)
        elif scheme == "Exact":
            # TODO: Correct this.
            for i in range(1,M+1):
                path_prev = np.maximum(path[i-1,:], 0) # Clip to have valid sqrt
                k = 4 * kappa * theta / (self.sigma**2)
                l = 4 * kappa * np.exp(-kappa * T_return[i]) / (self.sigma**2 * (1-np.exp(-kappa *T_return[i]))) * path[i-1]
                factor = self.sigma**2 * (1 - np.exp(-kappa * T_return[i])) / (4*kappa)
                path[i] = factor * ncx2.rvs(df = k, nc = l, random_state = seed)

        return T_return, path
    


    # MC pricing. 
    def get_cdso_pric_MC(self,params,t,t0,t_M,strikes,X0,N,M,seed=1000):
        # N prices are comuted and averaged MC
        N_strikes = strikes.shape[0]
        prices = np.zeros(shape = (N,N_strikes))
        prices_MC_hist =  np.zeros(shape = (N,N_strikes))
        for i in range(N):
            # Get default intensity process. 
            T_return,X_t = self.simulate_intensity(X0,t0,M,scheme = 'Milstein',seed=seed)
            # lambda_t is the sum of latent states..
            lambda_t = np.sum(X_t,axis=1)
            # Compute prob of default at time t0
            deltas = np.array([T_return[i]-T_return[i-1] for i in range(1,M+1)])
            Lambda = np.cumsum(lambda_t[1:]*deltas)
            # Determine if default or not at t0. If lambda>E\simEXPo(1) option payoff is zero.
            E = expon.rvs(random_state = seed)
            if Lambda[-1] >= E:
                prices[i] = 0
            # Else - begin to compute prices. 
            else: 
                X_t0 = X_t[-1,:]
                prot = self.calc_protection_leg(params, t0,t0, t_M, X_t0)
                # Quick fix due to way its written
                I1 = self.calc_coupon_leg(params,t0, t0, t_M, X_t0)
                I2 = self.calc_accrual_leg(params,t0, t0, t_M, X_t0)

                Value_CDS = prot - strikes * (I1 + I2)
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
        for i in range(N):
            # Get Latent states. Simulate path of CDS till mat.
            T_return,X_t = self.simulate_intensity(X0,t0,M,scheme = 'Milstein',seed=seed)
            # Compute prob of default at time t0
            lambda_t = np.sum(X_t,axis=1)
            # Compute prob of default at time t0
            deltas = np.array([T_return[i]-T_return[i-1] for i in range(1,M+1)])
            Lambda = np.cumsum(lambda_t[1:]*deltas)
            # Get model implies CDS.
            CDS_sim = np.zeros(Lambda.shape)
            for n in range(CDS_sim.shape[0]):    
                # CDS_sim[n] = self.cds_spread(X_t[n,:],params,T_return[n],
                #                              np.array([T_return[n] + t_M]),T_return[n]+t0)
                CDS_sim[n] = calc_cds(params, T_return[n], T_return[n] + t_M, 
                                      X_t[n,:], T_return[n]+t0, self.r, self.delta, 
                                      self.tenor, self.X_dim)
            # Determine if default or not at t0. If lambda>E\simEXPo(1) option payoff is zero.
            E = expon.rvs(random_state = seed)
            # Should be zero, but depends on barrier. Default happens at some point below..
            default_event =  Lambda >= E
            if np.any(default_event):
                # In this instance, default has happened as some point. Find index. 
                idx = np.argmax(np.where(default_event))
                # Get path maximum up to the point:
                max_cds_to_default = np.max(CDS_sim[:idx])
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
        for i in range(N):
            # Get Latent states. Simulate path of CDS till mat.
            T_return,X_t = self.simulate_intensity(X0,t0,M,scheme = 'Milstein',seed=seed)
            # Compute prob of default at time t0
            lambda_t = np.sum(X_t,axis=1)
            # Compute prob of default at time t0
            deltas = np.array([T_return[i]-T_return[i-1] for i in range(1,M+1)])
            Lambda = np.cumsum(lambda_t[1:]*deltas)
            # Get model implies CDS.
            CDS_sim = np.zeros(Lambda.shape)
            for n in range(CDS_sim.shape[0]):    
                # CDS_sim[n] = self.cds_spread(X_t[n,:],params,T_return[n],
                #                              np.array([T_return[n] + t_M]),T_return[n]+t0)
                CDS_sim[n] = calc_cds(params, T_return[n], T_return[n] + t_M, 
                                      X_t[n,:], T_return[n]+t0, self.r, self.delta, 
                                      self.tenor, self.X_dim)


            # Determine if default or not at t0. If lambda>E\simEXPo(1) option payoff is zero.
            E = expon.rvs(random_state = seed)
            # Should be zero, but depends on barrier. Default happens at some point below..
            default_event =  Lambda >= E
            if np.any(default_event):
                prices[i] = 0
                # set min to zero here
                cds_min[i] =0  # np.min(CDS_sim[:i])
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
